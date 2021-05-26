import time
import odrive
from odrive.enums import *

accel=20.
vel=5.
ratio=2
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

odrv0.axis0.requested_state=AXIS_STATE_CLOSED_LOOP_CONTROL
odrv0.axis1.requested_state=AXIS_STATE_CLOSED_LOOP_CONTROL

odrv0.axis0.controller.config.input_mode = INPUT_MODE_TRAP_TRAJ
odrv0.axis0.trap_traj.config.vel_limit=vel
odrv0.axis0.trap_traj.config.accel_limit=accel
odrv0.axis0.trap_traj.config.decel_limit=accel
odrv0.axis0.controller.config.inertia=0

pos0=odrv0.axis0.encoder.pos_estimate
pos1=odrv0.axis1.encoder.pos_estimate

odrv0.axis1.requested_state=AXIS_STATE_IDLE

while True:
    delta1=odrv0.axis1.encoder.pos_estimate-pos1
    odrv0.axis0.controller.input_pos=pos0+ratio*delta1
