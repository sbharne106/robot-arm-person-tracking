from adafruit_servokit import ServoKit
import time

kit = ServoKit(channels=16, address=0x40)

ELBOW_CHANNEL = 2

ELBOW_MIN = 100
ELBOW_MEAN = 240
ELBOW_MAX = 260

kit.servo[ELBOW_CHANNEL].set_pulse_width_range(500, 2500)
kit.servo[ELBOW_CHANNEL].actuation_range = 270

def move_elbow(offset):
    real_angle = ELBOW_MEAN + offset

    if real_angle < ELBOW_MIN:
        print(f"Too low. Clipping {real_angle} to {ELBOW_MIN}")
        real_angle = ELBOW_MIN

    if real_angle > ELBOW_MAX:
        print(f"Too high. Clipping {real_angle} to {ELBOW_MAX}")
        real_angle = ELBOW_MAX

    print(f"Elbow offset: {offset}, real servo angle: {real_angle}")
    kit.servo[ELBOW_CHANNEL].angle = real_angle
    time.sleep(1)

print("Elbow custom movement test")
print("0 = elbow mean/start position")
print("-140 = elbow min position")
print("+20 = elbow max position")
print("Type q to quit")

while True:
    user_input = input("Enter elbow offset -20 to 140, or q to quit: ")

    if user_input == "q":
        break

    offset = int(user_input)
    move_elbow(offset)

print("Done")
