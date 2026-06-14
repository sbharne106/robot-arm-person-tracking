from adafruit_servokit import ServoKit
import time

kit = ServoKit(channels=16, address=0x40)

ELBOW_CHANNEL = 2
BASE_CHANNEL = 3
SHOULDER_CHANNEL = 0

# Set up all servos
for channel in [BASE_CHANNEL, SHOULDER_CHANNEL, ELBOW_CHANNEL]:
    kit.servo[channel].set_pulse_width_range(500, 2500)
    kit.servo[channel].actuation_range = 270

def move_servo(channel, angle):
    angle = max(0, min(270, angle))  # keep angle between 0 and 180
    print(f"Moving channel {channel} to {angle} degrees")
    kit.servo[channel].angle = angle
    time.sleep(1)

print("Servo mean position finder")
print("We will test one servo at a time.")
print("Start with BASE servo.")

while True:
    user_input = input("Enter angle 0-270 for BASE, or q to quit: ")

    if user_input == "q":
        break

    angle = int(user_input)
    move_servo(BASE_CHANNEL, angle)

print("Done")
