import time
import odrive
from odrive.enums import *

accel=30.
vel=4.
calibration=False

odrv0=odrive.find_any(serial_number='205C3690424D')

if calibration:
    print("Calibration...", end='', flush=True)
    odrv0.axis0.requested_state=4
    odrv0.axis1.requested_state=4

    while odrv0.axis0.current_state != AXIS_STATE_IDLE:
        time.sleep(0.1)
    while odrv0.axis1.current_state != AXIS_STATE_IDLE:
        time.sleep(0.1)        
    print("OK")
    
odrv0.axis0.controller.config.input_mode=INPUT_MODE_TRAP_TRAJ
odrv0.axis0.trap_traj.config.vel_limit=vel
odrv0.axis0.trap_traj.config.accel_limit=accel
odrv0.axis0.trap_traj.config.decel_limit=accel
odrv0.axis0.controller.config.inertia=0

odrv0.axis1.controller.config.input_mode=INPUT_MODE_TRAP_TRAJ
odrv0.axis1.trap_traj.config.vel_limit=vel
odrv0.axis1.trap_traj.config.accel_limit=accel
odrv0.axis1.trap_traj.config.decel_limit=accel
odrv0.axis1.controller.config.inertia=0

odrv0.axis0.requested_state=AXIS_STATE_CLOSED_LOOP_CONTROL
odrv0.axis1.requested_state=AXIS_STATE_CLOSED_LOOP_CONTROL

pos0=odrv0.axis0.encoder.pos_estimate
pos1=odrv0.axis1.encoder.pos_estimate
shift=1/5

while True:
    pos0_finale=pos0+shift
    pos1_finale=pos1+shift
    odrv0.axis0.controller.input_pos=pos0_finale
    odrv0.axis1.controller.input_pos=pos1_finale
    while abs(odrv0.axis0.encoder.pos_estimate-pos0_finale)>0.02 or \
          abs(odrv0.axis1.encoder.pos_estimate-pos1_finale)>0.02:
        time.sleep(0.1)
    shift=-shift
