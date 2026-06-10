
from adafruit_servokit import ServoKit
import time

# -----------------------------
# PCA9685 setup
# -----------------------------
kit = ServoKit(channels=16, address=0x40)

# -----------------------------
# Servo channels
# Change these if your wiring is different
# -----------------------------
BASE_CHANNEL = 3
ELBOW_CHANNEL = 2
SHOULDER_CHANNEL = 0

# -----------------------------
# Calibration values
# -----------------------------

# Base calibration
BASE_MIN = 0
BASE_MEAN = 130
BASE_MAX = 270
BASE_DIRECTION = 1

# Elbow calibration
ELBOW_MIN = 100
ELBOW_MEAN = 240
ELBOW_MAX = 260
ELBOW_DIRECTION = -1   # reversed: positive moves toward real angle 100

# Shoulder calibration
SHOULDER_MIN = 30
SHOULDER_MEAN = 190
SHOULDER_MAX = 270
SHOULDER_DIRECTION = 1

# -----------------------------
# Servo setup
# -----------------------------
for channel in [BASE_CHANNEL, ELBOW_CHANNEL, SHOULDER_CHANNEL]:
    kit.servo[channel].set_pulse_width_range(500, 2500)
    kit.servo[channel].actuation_range = 270


def clamp(value, min_value, max_value):
    """Keep value inside safe min/max range."""
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def move_with_offset(name, channel, mean, min_angle, max_angle, offset, direction):
    """
    offset = 0 means mean position.
    positive/negative offset moves from mean.
    direction = 1 means normal.
    direction = -1 means reversed.
    """

    real_angle = mean + direction * offset
    real_angle = clamp(real_angle, min_angle, max_angle)

    print(f"{name}: offset = {offset}, real angle = {real_angle}")

    kit.servo[channel].angle = real_angle
    time.sleep(0.5)


def move_base(offset):
    move_with_offset(
        "BASE",
        BASE_CHANNEL,
        BASE_MEAN,
        BASE_MIN,
        BASE_MAX,
        offset,
        BASE_DIRECTION
    )


def move_elbow(offset):
    move_with_offset(
        "ELBOW",
        ELBOW_CHANNEL,
        ELBOW_MEAN,
        ELBOW_MIN,
        ELBOW_MAX,
        offset,
        ELBOW_DIRECTION
    )


def move_shoulder(offset):
    move_with_offset(
        "SHOULDER",
        SHOULDER_CHANNEL,
        SHOULDER_MEAN,
        SHOULDER_MIN,
        SHOULDER_MAX,
        offset,
        SHOULDER_DIRECTION
    )


def move_all_to_mean():
    print("Moving all servos to mean positions...")
    move_base(0)
    move_elbow(0)
    move_shoulder(0)


def show_ranges():
    print()
    print("Offset ranges:")
    print(f"Base:     {BASE_MIN - BASE_MEAN} to {BASE_MAX - BASE_MEAN}")
    print("Elbow:    because reversed:")
    print(f"          positive max = {ELBOW_MEAN - ELBOW_MIN}")
    print(f"          negative max = {ELBOW_MEAN - ELBOW_MAX}")
    print(f"Shoulder: {SHOULDER_MIN - SHOULDER_MEAN} to {SHOULDER_MAX - SHOULDER_MEAN}")
    print()


# -----------------------------
# Main menu
# -----------------------------
print("Robot arm 3-servo offset control")
print("--------------------------------")
print("0 offset = mean position")
print("Positive/negative offset = move from mean")
print()
print("Commands:")
print("b = move base")
print("e = move elbow")
print("s = move shoulder")
print("a = move all with offsets")
print("m = move all to mean")
print("r = show offset ranges")
print("q = quit")
print()

move_all_to_mean()
show_ranges()

while True:
    command = input("Choose command b/e/s/a/m/r/q: ").lower()

    if command == "q":
        print("Quitting safely...")
        move_all_to_mean()
        time.sleep(1)

        kit.servo[BASE_CHANNEL].angle = None
        kit.servo[ELBOW_CHANNEL].angle = None
        kit.servo[SHOULDER_CHANNEL].angle = None

        print("Motors moved to mean and servo signals cut off.")
        break

    elif command == "m":
        move_all_to_mean()

    elif command == "r":
        show_ranges()

    elif command == "b":
        offset = int(input("Enter BASE offset: "))
        move_base(offset)

    elif command == "e":
        offset = int(input("Enter ELBOW offset: "))
        move_elbow(offset)

    elif command == "s":
        offset = int(input("Enter SHOULDER offset: "))
        move_shoulder(offset)

    elif command == "a":
        base_offset = int(input("Enter BASE offset: "))
        elbow_offset = int(input("Enter ELBOW offset: "))
        shoulder_offset = int(input("Enter SHOULDER offset: "))

        move_base(base_offset)
        move_elbow(elbow_offset)
        move_shoulder(shoulder_offset)

    else:
        print("Invalid command. Use b, e, s, a, m, r, or q.")
