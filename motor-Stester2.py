from adafruit_servokit import ServoKit
import time

kit = ServoKit(channels=16, address=0x40)

SHOULDER_CHANNEL = 0

SHOULDER_MIN = 30
SHOULDER_MEAN = 190
SHOULDER_MAX = 270
SHOULDER_DIRECTION = 1   # change to -1 if you want shoulder reversed

kit.servo[SHOULDER_CHANNEL].set_pulse_width_range(500, 2500)
kit.servo[SHOULDER_CHANNEL].actuation_range = 270

def clamp(value, min_value, max_value):
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value

def move_shoulder(offset):
    real_angle = SHOULDER_MEAN + SHOULDER_DIRECTION * offset
    real_angle = clamp(real_angle, SHOULDER_MIN, SHOULDER_MAX)

    print(f"Shoulder offset: {offset}, real servo angle: {real_angle}")
    kit.servo[SHOULDER_CHANNEL].angle = real_angle
    time.sleep(1)

print("Shoulder offset movement test")
print("0 = shoulder mean position")
print("-160 = shoulder min position")
print("+80 = shoulder max position")
print("Type q to quit")

while True:
    user_input = input("Enter shoulder offset -160 to +80, or q to quit: ")

    if user_input == "q":
        break

    try:
        offset = int(user_input)
        move_shoulder(offset)
    except ValueError:
        print("Please enter a number or q.")

print("Done")
