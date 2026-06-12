from adafruit_servokit import ServoKit
import time

# -----------------------------
# PCA9685 setup
# -----------------------------
kit = ServoKit(channels=16, address=0x40)

# -----------------------------
# Servo channels
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
ELBOW_DIRECTION = -1   # positive offset moves toward real angle 100

# Shoulder calibration
SHOULDER_MIN = 30
SHOULDER_MEAN = 190
SHOULDER_MAX = 270
SHOULDER_DIRECTION = 1

# -----------------------------
# Coordinated movement settings
# -----------------------------
MOVE_STEPS = 60
MOVE_DELAY = 0.02

# -----------------------------
# Servo setup
# -----------------------------
for channel in [BASE_CHANNEL, ELBOW_CHANNEL, SHOULDER_CHANNEL]:
    kit.servo[channel].set_pulse_width_range(500, 2500)
    kit.servo[channel].actuation_range = 270


# -----------------------------
# Current offsets
# 0 offset = mean position
# -----------------------------
current_base_offset = 0
current_elbow_offset = 0
current_shoulder_offset = 0


def clamp(value, min_value, max_value):
    """Keep value inside min/max range."""
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def offset_to_real(mean, offset, direction):
    """
    Convert offset value to real servo angle.
    offset = 0 means mean position.
    """
    return mean + direction * offset


def real_to_safe_offset_limits(mean, min_angle, max_angle, direction):
    """
    Calculate safe offset range based on real servo min/max.
    Works even if direction is reversed.
    """
    offset1 = (min_angle - mean) / direction
    offset2 = (max_angle - mean) / direction

    return min(offset1, offset2), max(offset1, offset2)


# -----------------------------
# Offset ranges from calibration
# -----------------------------
BASE_OFFSET_MIN, BASE_OFFSET_MAX = real_to_safe_offset_limits(
    BASE_MEAN,
    BASE_MIN,
    BASE_MAX,
    BASE_DIRECTION
)

ELBOW_OFFSET_MIN, ELBOW_OFFSET_MAX = real_to_safe_offset_limits(
    ELBOW_MEAN,
    ELBOW_MIN,
    ELBOW_MAX,
    ELBOW_DIRECTION
)

SHOULDER_OFFSET_MIN, SHOULDER_OFFSET_MAX = real_to_safe_offset_limits(
    SHOULDER_MEAN,
    SHOULDER_MIN,
    SHOULDER_MAX,
    SHOULDER_DIRECTION
)


# -----------------------------
# Safety rules
# -----------------------------
# Rule 1:
# When shoulder is -160,
# elbow should only be allowed from 120 to 140.
SHOULDER_DANGER_OFFSET = -160
SHOULDER_DANGER_ELBOW_MIN = 120
SHOULDER_DANGER_ELBOW_MAX = 140

# Rule 2:
# When base is -130,
# elbow should only be allowed from 130 to 140,
# shoulder should only be allowed from -100 to 80.
BASE_DANGER_OFFSET = -130
BASE_DANGER_ELBOW_MIN = 130
BASE_DANGER_ELBOW_MAX = 140
BASE_DANGER_SHOULDER_MIN = -100
BASE_DANGER_SHOULDER_MAX = 80


def apply_safety_rules(base_offset, elbow_offset, shoulder_offset):
    """
    Applies general motor limits and special camera-damage safety rules.

    Input and output are offset values, not real servo angles.
    """

    # First clamp to general calibrated ranges
    base_offset = clamp(base_offset, BASE_OFFSET_MIN, BASE_OFFSET_MAX)
    elbow_offset = clamp(elbow_offset, ELBOW_OFFSET_MIN, ELBOW_OFFSET_MAX)
    shoulder_offset = clamp(shoulder_offset, SHOULDER_OFFSET_MIN, SHOULDER_OFFSET_MAX)

    # -------------------------------------------------
    # Safety Rule 2: base at -130 danger zone
    # This has higher priority.
    # -------------------------------------------------
    if base_offset <= BASE_DANGER_OFFSET:
        shoulder_offset = clamp(
            shoulder_offset,
            BASE_DANGER_SHOULDER_MIN,
            BASE_DANGER_SHOULDER_MAX
        )

        elbow_offset = clamp(
            elbow_offset,
            BASE_DANGER_ELBOW_MIN,
            BASE_DANGER_ELBOW_MAX
        )

    # -------------------------------------------------
    # Safety Rule 1: shoulder at -160 danger zone
    # -------------------------------------------------
    if shoulder_offset <= SHOULDER_DANGER_OFFSET:
        elbow_offset = clamp(
            elbow_offset,
            SHOULDER_DANGER_ELBOW_MIN,
            SHOULDER_DANGER_ELBOW_MAX
        )

    return base_offset, elbow_offset, shoulder_offset


def write_all_real_angles(base_offset, elbow_offset, shoulder_offset):
    """
    Converts offset values to real servo angles and writes to all motors.
    """

    base_real = offset_to_real(BASE_MEAN, base_offset, BASE_DIRECTION)
    elbow_real = offset_to_real(ELBOW_MEAN, elbow_offset, ELBOW_DIRECTION)
    shoulder_real = offset_to_real(SHOULDER_MEAN, shoulder_offset, SHOULDER_DIRECTION)

    base_real = clamp(base_real, BASE_MIN, BASE_MAX)
    elbow_real = clamp(elbow_real, ELBOW_MIN, ELBOW_MAX)
    shoulder_real = clamp(shoulder_real, SHOULDER_MIN, SHOULDER_MAX)

    kit.servo[BASE_CHANNEL].angle = base_real
    kit.servo[ELBOW_CHANNEL].angle = elbow_real
    kit.servo[SHOULDER_CHANNEL].angle = shoulder_real

    return base_real, elbow_real, shoulder_real


def print_pose(label, base_offset, elbow_offset, shoulder_offset):
    base_real = offset_to_real(BASE_MEAN, base_offset, BASE_DIRECTION)
    elbow_real = offset_to_real(ELBOW_MEAN, elbow_offset, ELBOW_DIRECTION)
    shoulder_real = offset_to_real(SHOULDER_MEAN, shoulder_offset, SHOULDER_DIRECTION)

    print()
    print(label)
    print(f"BASE     offset = {base_offset:.1f}, real angle = {base_real:.1f}")
    print(f"ELBOW    offset = {elbow_offset:.1f}, real angle = {elbow_real:.1f}")
    print(f"SHOULDER offset = {shoulder_offset:.1f}, real angle = {shoulder_real:.1f}")


def move_all_offsets(base_target, elbow_target, shoulder_target):
    """
    Main coordinated movement function.

    Every motor moves at the same time.
    Safety is checked:
    1. at the final target
    2. during every small movement step

    This prevents the arm from passing through unsafe positions.
    """

    global current_base_offset
    global current_elbow_offset
    global current_shoulder_offset

    print_pose(
        "Requested target:",
        base_target,
        elbow_target,
        shoulder_target
    )

    safe_base_target, safe_elbow_target, safe_shoulder_target = apply_safety_rules(
        base_target,
        elbow_target,
        shoulder_target
    )

    print_pose(
        "Safe target after rules:",
        safe_base_target,
        safe_elbow_target,
        safe_shoulder_target
    )

    start_base = current_base_offset
    start_elbow = current_elbow_offset
    start_shoulder = current_shoulder_offset

    for step in range(1, MOVE_STEPS + 1):
        fraction = step / MOVE_STEPS

        # Planned coordinated position for this step
        base_now = start_base + (safe_base_target - start_base) * fraction
        elbow_now = start_elbow + (safe_elbow_target - start_elbow) * fraction
        shoulder_now = start_shoulder + (safe_shoulder_target - start_shoulder) * fraction

        # Safety check at every step
        base_now, elbow_now, shoulder_now = apply_safety_rules(
            base_now,
            elbow_now,
            shoulder_now
        )

        write_all_real_angles(base_now, elbow_now, shoulder_now)
        time.sleep(MOVE_DELAY)

    current_base_offset = safe_base_target
    current_elbow_offset = safe_elbow_target
    current_shoulder_offset = safe_shoulder_target

    print_pose(
        "Movement complete:",
        current_base_offset,
        current_elbow_offset,
        current_shoulder_offset
    )


def move_base(offset):
    """
    Move base while elbow and shoulder stay coordinated/safe.
    """
    move_all_offsets(
        offset,
        current_elbow_offset,
        current_shoulder_offset
    )


def move_elbow(offset):
    """
    Move elbow while base and shoulder stay coordinated/safe.
    """
    move_all_offsets(
        current_base_offset,
        offset,
        current_shoulder_offset
    )


def move_shoulder(offset):
    """
    Move shoulder while base and elbow stay coordinated/safe.
    """
    move_all_offsets(
        current_base_offset,
        current_elbow_offset,
        offset
    )


def move_all_to_mean():
    """
    Move all motors back to offset 0 together.
    """
    print("Moving all servos to mean positions...")
    move_all_offsets(0, 0, 0)


def cut_off_servos():
    kit.servo[BASE_CHANNEL].angle = None
    kit.servo[ELBOW_CHANNEL].angle = None
    kit.servo[SHOULDER_CHANNEL].angle = None


def show_ranges():
    print()
    print("Offset ranges from calibration:")
    print(f"Base:     {BASE_OFFSET_MIN:.0f} to {BASE_OFFSET_MAX:.0f}")
    print(f"Elbow:    {ELBOW_OFFSET_MIN:.0f} to {ELBOW_OFFSET_MAX:.0f}")
    print(f"Shoulder: {SHOULDER_OFFSET_MIN:.0f} to {SHOULDER_OFFSET_MAX:.0f}")
    print()
    print("Special safety rules:")
    print("1. If shoulder reaches -160:")
    print("   elbow is limited to 120 to 140")
    print()
    print("2. If base reaches -130:")
    print("   elbow is limited to 130 to 140")
    print("   shoulder is limited to -100 to 80")
    print()


# -----------------------------
# Main menu
# -----------------------------
print("Robot arm 3-servo coordinated offset control")
print("--------------------------------------------")
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
print("t1 = test shoulder safety")
print("t2 = test base safety")
print("q = quit")
print()

move_all_to_mean()
show_ranges()

while True:
    command = input("Choose command b/e/s/a/m/r/t1/t2/q: ").lower().strip()

    if command == "q":
        print("Quitting safely...")
        move_all_to_mean()
        time.sleep(1)
        cut_off_servos()
        print("Motors moved to mean and servo signals cut off.")
        break

    elif command == "m":
        move_all_to_mean()

    elif command == "r":
        show_ranges()

    elif command == "b":
        try:
            offset = int(input("Enter BASE offset: "))
            move_base(offset)
        except ValueError:
            print("Invalid input. Enter a number.")

    elif command == "e":
        try:
            offset = int(input("Enter ELBOW offset: "))
            move_elbow(offset)
        except ValueError:
            print("Invalid input. Enter a number.")

    elif command == "s":
        try:
            offset = int(input("Enter SHOULDER offset: "))
            move_shoulder(offset)
        except ValueError:
            print("Invalid input. Enter a number.")

    elif command == "a":
        try:
            base_offset = int(input("Enter BASE offset: "))
            elbow_offset = int(input("Enter ELBOW offset: "))
            shoulder_offset = int(input("Enter SHOULDER offset: "))

            move_all_offsets(
                base_offset,
                elbow_offset,
                shoulder_offset
            )

        except ValueError:
            print("Invalid input. Enter numbers only.")

    elif command == "t1":
        print("Testing condition 1:")
        print("Trying shoulder = -160 and elbow = 100")
        print("Expected: elbow should be forced to 120 minimum.")
        move_all_offsets(
            0,
            100,
            -160
        )

    elif command == "t2":
        print("Testing condition 2:")
        print("Trying base = -130, elbow = 100, shoulder = -160")
        print("Expected: elbow forced to 130 minimum and shoulder forced to -100 minimum.")
        move_all_offsets(
            -130,
            100,
            -160
        )

    else:
        print("Invalid command. Use b, e, s, a, m, r, t1, t2, or q.")
