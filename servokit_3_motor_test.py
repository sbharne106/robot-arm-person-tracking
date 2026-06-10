from adafruit_servokit import ServoKit
import time

# PCA9685 default address is 0x40
kit = ServoKit(channels=16, address=0x40)

BASE_CHANNEL = 5
MOTOR_1_CHANNEL = 3
MOTOR_2_CHANNEL = 4

# Set pulse range and angle range for all three servos
for channel in [BASE_CHANNEL, MOTOR_1_CHANNEL, MOTOR_2_CHANNEL]:
    kit.servo[channel].set_pulse_width_range(500, 2500)
    kit.servo[channel].actuation_range = 180

print("Starting 3 motor/servo test")

# Move all to center
print("Moving all to 90")
kit.servo[BASE_CHANNEL].angle = 90
kit.servo[MOTOR_1_CHANNEL].angle = 90
kit.servo[MOTOR_2_CHANNEL].angle = 90
time.sleep(1)

# Test base servo
print("Testing base servo on channel 5")
kit.servo[BASE_CHANNEL].angle = 60
time.sleep(1)
kit.servo[BASE_CHANNEL].angle = 120
time.sleep(1)
kit.servo[BASE_CHANNEL].angle = 90
time.sleep(1)

# Test motor/servo on channel 3
print("Testing motor/servo on channel 3")
kit.servo[MOTOR_1_CHANNEL].angle = 60
time.sleep(1)
kit.servo[MOTOR_1_CHANNEL].angle = 120
time.sleep(1)
kit.servo[MOTOR_1_CHANNEL].angle = 90
time.sleep(1)

# Test motor/servo on channel 4
print("Testing motor/servo on channel 4")
kit.servo[MOTOR_2_CHANNEL].angle = 60
time.sleep(1)
kit.servo[MOTOR_2_CHANNEL].angle = 120
time.sleep(1)
kit.servo[MOTOR_2_CHANNEL].angle = 90
time.sleep(1)

# Move all together
print("Moving all together")
kit.servo[BASE_CHANNEL].angle = 60
kit.servo[MOTOR_1_CHANNEL].angle = 60
kit.servo[MOTOR_2_CHANNEL].angle = 60
time.sleep(1)

kit.servo[BASE_CHANNEL].angle = 120
kit.servo[MOTOR_1_CHANNEL].angle = 120
kit.servo[MOTOR_2_CHANNEL].angle = 120
time.sleep(1)

kit.servo[BASE_CHANNEL].angle = 90
kit.servo[MOTOR_1_CHANNEL].angle = 90
kit.servo[MOTOR_2_CHANNEL].angle = 90
time.sleep(1)

print("Done")
