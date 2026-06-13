 
from adafruit_servokit import ServoKit
import time


# PCA9685 default address is 0x40
kit = ServoKit(channels=16, address=0x40)

ELBOW_CHANNEL = 2
BASE_CHANNEL = 3
SHOULDER_CHANNEL = 4

# Set pulse range and angle range for all three servos
for channel in [BASE_CHANNEL, SHOULDER_CHANNEL, ELBOW_CHANNEL]:
    kit.servo[channel].set_pulse_width_range(500, 2500)
    kit.servo[channel].actuation_range = 180

print("Starting 3 motor/servo test")

#set up 
# Move all to center
print("Moving all to 90")
kit.servo[BASE_CHANNEL].angle = 60
time.sleep(1)

kit.servo[BASE_CHANNEL].angle = 90
time.sleep(1)

kit.servo[BASE_CHANNEL].angle = 120
time.sleep(1)

#kit.servo[BASE_CHANNEL].angle = 0
#time.sleep(1)

#kit.servo[BASE_CHANNEL].angle = 90
#time.sleep(1)

#kit.servo[BASE_CHANNEL].angle = 0
#time.sleep(1)
#kit.servo[BASE_CHANNEL].angle = 0
#kit.servo[BASE_CHANNEL].angle = 90
#kit.servo[BASE_CHANNEL].angle = 180
#kit.servo[SHOULDER_CHANNEL].angle = 90
#kit.servo[ELBOW_CHANNEL].angle = 90
time.sleep(1)
