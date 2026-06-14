from adafruit_servokit import ServoKit
import time

kit = ServoKit(channels=16, address=0x40)

BASE_CHANNEL = 3   # change if your base is wired to another channel

BASE_MEAN = 130
BASE_MIN = 0
BASE_MAX = 270

kit.servo[BASE_CHANNEL].set_pulse_width_range(500, 2500)
kit.servo[BASE_CHANNEL].actuation_range = BASE_MAX

def move_base_custom(offset):
    """
    offset is movement from your mean position.
    offset = 0 means mean position.
    offset = -130 means real servo angle 0.
    offset = 140 means real servo angle 270.
    """

    real_angle = BASE_MEAN + offset

    # keep inside safe range
    if real_angle < BASE_MIN:
        print(f"Too low. Clipping {real_angle} to {BASE_MIN}")
        real_angle = BASE_MIN

    if real_angle > BASE_MAX:
        print(f"Too high. Clipping {real_angle} to {BASE_MAX}")
        real_angle = BASE_MAX

    print(f"Custom offset: {offset}, real servo angle: {real_angle}")
    kit.servo[BASE_CHANNEL].angle = real_angle
    time.sleep(1)

print("Base custom movement test")
print("0 = mean position")
print("-130 = one side")
print("+140 = other side")

while True:
    user_input = input("Enter custom offset -130 to 140, or q to quit: ")

    if user_input == "q":
        break

    offset = int(user_input)
    move_base_custom(offset)

print("Done")
