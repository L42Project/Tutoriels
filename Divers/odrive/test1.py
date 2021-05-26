import time
import odrive
from odrive.enums import *

accel=10.
vel=3.
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

odrv0.axis0.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
odrv0.axis0.controller.config.input_mode = INPUT_MODE_VEL_RAMP
odrv0.axis1.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
odrv0.axis1.controller.config.input_mode = INPUT_MODE_VEL_RAMP
    
odrv0.axis0.controller.config.vel_ramp_rate=accel
odrv0.axis1.controller.config.vel_ramp_rate=accel

odrv0.axis0.requested_state=AXIS_STATE_CLOSED_LOOP_CONTROL
odrv0.axis1.requested_state=AXIS_STATE_CLOSED_LOOP_CONTROL

odrv0.axis0.controller.input_vel=vel
odrv0.axis1.controller.input_vel=0

evenement=time.time()
id=0
while True:
    if id==0:
        torque=odrv0.axis0.motor.current_control.Iq_setpoint*odrv0.axis0.motor.config.torque_constant
    else:
        torque=odrv0.axis1.motor.current_control.Iq_setpoint*odrv0.axis1.motor.config.torque_constant
    if abs(torque)>0.2 and (time.time()-evenement)>1:
        if id==0:
            odrv0.axis0.controller.input_vel=0
            odrv0.axis1.controller.input_vel=vel
            id=1
        else:
            odrv0.axis1.controller.input_vel=0
            odrv0.axis0.controller.input_vel=vel
            id=0
        evenement0=time.time()
            

