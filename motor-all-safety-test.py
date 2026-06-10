import time
from adafruit_servokit import ServoKit

# ============================================================
# PCA9685 SETUP
# ============================================================
kit = ServoKit(channels=16)

# ============================================================
# SERVO CHANNELS
# Change these if your wiring is different
# ============================================================
BASE_CHANNEL = 3
SHOULDER_CHANNEL = 0
ELBOW_CHANNEL = 2
# ============================================================
# SERVO ACTUATION RANGES
# ============================================================
kit.servo[BASE_CHANNEL].actuation_range = 270
kit.servo[SHOULDER_CHANNEL].actuation_range = 270
kit.servo[ELBOW_CHANNEL].actuation_range = 270

# ============================================================
# OFFSET SYSTEM CALIBRATION
# ============================================================
# Logic value 0 means the motor is at its mean position.
#
# Raw servo angle = MEAN_RAW + logic_offset
#
# Example:
# shoulder mean raw = 190
# shoulder logic = -160
# actual servo angle = 190 + (-160) = 30

BASE_MEAN_RAW = 190       # change if your base mean is different
SHOULDER_MEAN_RAW = 190   # your tested shoulder mean
ELBOW_MEAN_RAW = 140      # change if your elbow mean is different

# ============================================================
# GENERAL SAFE LOGIC LIMITS
# These are the normal allowed offset ranges before special rules.
# ============================================================

BASE_MIN_LOGIC = -130
BASE_MAX_LOGIC = 80

SHOULDER_MIN_LOGIC = -160
SHOULDER_MAX_LOGIC = 80

ELBOW_MIN_LOGIC = 100
ELBOW_MAX_LOGIC = 160

# ============================================================
# SPECIAL SAFETY RULES
# ============================================================
# Rule 1:
# When shoulder reaches -160,
# elbow must only stay between 120 and 140.
#
# Rule 2:
# When base reaches -130,
# elbow must only stay between 130 and 140,
# shoulder must only stay between -100 and 80.

SHOULDER_DANGER_POSITION = -160
SHOULDER_DANGER_ELBOW_MIN = 120
SHOULDER_DANGER_ELBOW_MAX = 140

BASE_DANGER_POSITION = -130
BASE_DANGER_ELBOW_MIN = 130
BASE_DANGER_ELBOW_MAX = 140
BASE_DANGER_SHOULDER_MIN = -100
BASE_DANGER_SHOULDER_MAX = 80

# ============================================================
# MOTION SETTINGS
# ============================================================
MOVE_STEPS = 60
MOVE_DELAY = 0.02

# Current logic positions
current_base = 0
current_shoulder = 0
current_elbow = 0


# ============================================================
# BASIC FUNCTIONS
# ============================================================
def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def logic_to_raw(logic_value, mean_raw):
    return mean_raw + logic_value


def write_motor_logic(channel, logic_value, mean_raw):
    raw_angle = logic_to_raw(logic_value, mean_raw)
    kit.servo[channel].angle = raw_angle


def print_pose(label, base, shoulder, elbow):
    print()
    print(label)
    print(f"Base logic:     {base}")
    print(f"Shoulder logic: {shoulder}")
    print(f"Elbow logic:    {elbow}")
    print(f"Base raw:       {logic_to_raw(base, BASE_MEAN_RAW)}")
    print(f"Shoulder raw:   {logic_to_raw(shoulder, SHOULDER_MEAN_RAW)}")
    print(f"Elbow raw:      {logic_to_raw(elbow, ELBOW_MEAN_RAW)}")


# ============================================================
# SAFETY LOGIC
# ============================================================
def apply_safety_rules(base_target, shoulder_target, elbow_target):
    """
    Takes requested target positions and modifies them if needed
    so the arm never enters unsafe combinations.
    """

    # First clamp to general ranges
    base_safe = clamp(base_target, BASE_MIN_LOGIC, BASE_MAX_LOGIC)
    shoulder_safe = clamp(shoulder_target, SHOULDER_MIN_LOGIC, SHOULDER_MAX_LOGIC)
    elbow_safe = clamp(elbow_target, ELBOW_MIN_LOGIC, ELBOW_MAX_LOGIC)

    # --------------------------------------------------------
    # RULE 2 has higher priority:
    # If base is at -130, shoulder cannot go beyond -100,
    # and elbow must stay between 130 and 140.
    # --------------------------------------------------------
    if base_safe <= BASE_DANGER_POSITION:
        shoulder_safe = clamp(
            shoulder_safe,
            BASE_DANGER_SHOULDER_MIN,
            BASE_DANGER_SHOULDER_MAX
        )

        elbow_safe = clamp(
            elbow_safe,
            BASE_DANGER_ELBOW_MIN,
            BASE_DANGER_ELBOW_MAX
        )

    # --------------------------------------------------------
    # RULE 1:
    # If shoulder is at -160, elbow must stay between 120 and 140.
    # This only applies if base danger rule did not already force
    # shoulder away from -160.
    # --------------------------------------------------------
    if shoulder_safe <= SHOULDER_DANGER_POSITION:
        elbow_safe = clamp(
            elbow_safe,
            SHOULDER_DANGER_ELBOW_MIN,
            SHOULDER_DANGER_ELBOW_MAX
        )

    return base_safe, shoulder_safe, elbow_safe


def get_safe_intermediate_pose(base, shoulder, elbow):
    """
    This is used at every small step during movement.
    It makes sure even the path between start and end is safe,
    not just the final position.
    """
    return apply_safety_rules(base, shoulder, elbow)


# ============================================================
# COORDINATED MOVEMENT FUNCTION
# ============================================================
def move_all_coordinated(base_target, shoulder_target, elbow_target):
    """
    Moves base, shoulder, and elbow together.

    The important part:
    during every small step, safety rules are applied.
    So if shoulder is moving toward -160, the elbow is also
    adjusted along the way if needed.
    """

    global current_base, current_shoulder, current_elbow

    # Apply safety to final target first
    safe_base_target, safe_shoulder_target, safe_elbow_target = apply_safety_rules(
        base_target,
        shoulder_target,
        elbow_target
    )

    print_pose(
        "Requested target:",
        base_target,
        shoulder_target,
        elbow_target
    )

    print_pose(
        "Safe target after rules:",
        safe_base_target,
        safe_shoulder_target,
        safe_elbow_target
    )

    start_base = current_base
    start_shoulder = current_shoulder
    start_elbow = current_elbow

    for step in range(1, MOVE_STEPS + 1):
        fraction = step / MOVE_STEPS

        # Planned interpolated positions
        base_now = start_base + (safe_base_target - start_base) * fraction
        shoulder_now = start_shoulder + (safe_shoulder_target - start_shoulder) * fraction
        elbow_now = start_elbow + (safe_elbow_target - start_elbow) * fraction

        # Apply safety at every step
        base_now, shoulder_now, elbow_now = get_safe_intermediate_pose(
            base_now,
            shoulder_now,
            elbow_now
        )

        # Send to motors
        write_motor_logic(BASE_CHANNEL, base_now, BASE_MEAN_RAW)
        write_motor_logic(SHOULDER_CHANNEL, shoulder_now, SHOULDER_MEAN_RAW)
        write_motor_logic(ELBOW_CHANNEL, elbow_now, ELBOW_MEAN_RAW)

        time.sleep(MOVE_DELAY)

    current_base = safe_base_target
    current_shoulder = safe_shoulder_target
    current_elbow = safe_elbow_target

    print_pose(
        "Movement complete at:",
        current_base,
        current_shoulder,
        current_elbow
    )


# ============================================================
# HOME / SHUTDOWN FUNCTIONS
# ============================================================
def move_to_mean():
    print("Moving all motors to mean position...")
    move_all_coordinated(0, 0, 0)


def cut_off_servos():
    print("Cutting off servo signals...")
    kit.servo[BASE_CHANNEL].angle = None
    kit.servo[SHOULDER_CHANNEL].angle = None
    kit.servo[ELBOW_CHANNEL].angle = None


# ============================================================
# MAIN TEST MENU
# ============================================================
print("Motor safety coordination test started.")
print("Using offset logic values.")
print()
print("Commands:")
print("1 = Test shoulder danger: shoulder -160, elbow tries 100")
print("2 = Test base danger: base -130, shoulder tries -160, elbow tries 100")
print("3 = Safe normal pose: base 0, shoulder -80, elbow 140")
print("4 = Move to mean: base 0, shoulder 0, elbow 0")
print("m = manually enter base shoulder elbow")
print("q = quit safely")
print()

try:
    move_to_mean()

    while True:
        command = input("\nEnter command: ").strip().lower()

        if command == "1":
            # Shoulder wants -160.
            # Elbow tries to go below 120.
            # Code should force elbow to 120 minimum.
            move_all_coordinated(
                base_target=0,
                shoulder_target=-160,
                elbow_target=100
            )

        elif command == "2":
            # Base wants -130.
            # Shoulder tries -160 but should be limited to -100.
            # Elbow tries 100 but should be limited to 130.
            move_all_coordinated(
                base_target=-130,
                shoulder_target=-160,
                elbow_target=100
            )

        elif command == "3":
            move_all_coordinated(
                base_target=0,
                shoulder_target=-80,
                elbow_target=140
            )

        elif command == "4":
            move_to_mean()

        elif command == "m":
            try:
                base_input = float(input("Enter base logic target: "))
                shoulder_input = float(input("Enter shoulder logic target: "))
                elbow_input = float(input("Enter elbow logic target: "))

                move_all_coordinated(
                    base_target=base_input,
                    shoulder_target=shoulder_input,
                    elbow_target=elbow_input
                )

            except ValueError:
                print("Invalid input. Enter numbers only.")

        elif command == "q":
            print("Quit requested.")
            break

        else:
            print("Unknown command.")

except KeyboardInterrupt:
    print("\nProgram interrupted.")

finally:
    move_to_mean()
    time.sleep(1)
    cut_off_servos()
    print("All motors returned to mean and signals cut off.")
