from adafruit_servokit import ServoKit
import time

kit = ServoKit(channels=16, address=0x40)

# Change this if your shoulder is on a different PCA9685 channel
SHOULDER_CHANNEL = 0

SHOULDER_MIN = 0
SHOULDER_MAX = 270

kit.servo[SHOULDER_CHANNEL].set_pulse_width_range(500, 2500)
kit.servo[SHOULDER_CHANNEL].actuation_range = 270

def move_shoulder(angle):
    if angle < SHOULDER_MIN:
        print(f"Too low. Clipping {angle} to {SHOULDER_MIN}")
        angle = SHOULDER_MIN

    if angle > SHOULDER_MAX:
        print(f"Too high. Clipping {angle} to {SHOULDER_MAX}")
        angle = SHOULDER_MAX

    print(f"Moving SHOULDER to real servo angle: {angle}")
    kit.servo[SHOULDER_CHANNEL].angle = angle
    time.sleep(1)

print("Shoulder mean/min/max finder")
print("Enter real servo angles from 0 to 270.")
print("Find:")
print("1. The angle you want as SHOULDER_MEAN")
print("2. The lowest safe angle as SHOULDER_MIN")
print("3. The highest safe angle as SHOULDER_MAX")
print("Type q to quit.")

while True:
    user_input = input("Enter SHOULDER angle 0-270, or q to quit: ")

    if user_input == "q":
        break

    try:
        angle = int(user_input)
        move_shoulder(angle)
    except ValueError:
        print("Please enter a number or q.")

print("Done")
