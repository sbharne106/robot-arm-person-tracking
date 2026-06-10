from adafruit_servokit import ServoKit
import time

kit = ServoKit(channels=16, address=0x40)

# Channels
ELBOW_CHANNEL = 2
BASE_CHANNEL = 3
SHOULDER_CHANNEL = 4

# Elbow settings
ELBOW_MIN = 60
ELBOW_MAX = 260
ELBOW_MEAN = 180

# Set up elbow servo
kit.servo[ELBOW_CHANNEL].set_pulse_width_range(500, 2500)
kit.servo[ELBOW_CHANNEL].actuation_range = ELBOW_MAX

def move_elbow(angle):
    if angle < ELBOW_MIN:
        print(f"Too low. Clipping {angle} to {ELBOW_MIN}")
        angle = ELBOW_MIN

    if angle > ELBOW_MAX:
        print(f"Too high. Clipping {angle} to {ELBOW_MAX}")
        angle = ELBOW_MAX

    print(f"Moving ELBOW to real servo angle: {angle}")
    kit.servo[ELBOW_CHANNEL].angle = angle
    time.sleep(1)

print("Elbow mean position finder")
print("Enter real servo angles from 0 to 270.")
print("Find the angle where the elbow is in the position you want to call mean.")
print("Type q to quit.")

while True:
    user_input = input("Enter ELBOW angle 0-270, or q to quit: ")

    if user_input == "q":
        break

    angle = int(user_input)
    move_elbow(angle)

print("Done")
