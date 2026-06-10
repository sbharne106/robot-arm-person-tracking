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
BASE_MIN = 0
BASE_MEAN = 130
BASE_MAX = 270

ELBOW_MIN = 100
ELBOW_MEAN = 240
ELBOW_MAX = 260

SHOULDER_MIN = 30
SHOULDER_MEAN = 190
SHOULDER_MAX = 270

# -----------------------------
# Servo setup
# -----------------------------
for channel in [BASE_CHANNEL, ELBOW_CHANNEL, SHOULDER_CHANNEL]:
    kit.servo[channel].set_pulse_width_range(500, 2500)
    kit.servo[channel].actuation_range = 270


def move_servo(name, channel, angle, delay=1.0):
    print(f"{name}: moving to {angle}")
    kit.servo[channel].angle = angle
    time.sleep(delay)


def move_all_to_mean():
    print("Moving all motors to mean positions...")

    kit.servo[BASE_CHANNEL].angle = BASE_MEAN
    kit.servo[ELBOW_CHANNEL].angle = ELBOW_MEAN
    kit.servo[SHOULDER_CHANNEL].angle = SHOULDER_MEAN

    time.sleep(1)
    print("All motors are at mean position.")


def move_all_to_max():
    print("\nMoving ALL motors to MAX positions together...")

    kit.servo[BASE_CHANNEL].angle = BASE_MAX
    kit.servo[ELBOW_CHANNEL].angle = ELBOW_MAX
    kit.servo[SHOULDER_CHANNEL].angle = SHOULDER_MAX

    time.sleep(1)


def move_all_to_min():
    print("\nMoving ALL motors to MIN positions together...")

    kit.servo[BASE_CHANNEL].angle = BASE_MIN
    kit.servo[ELBOW_CHANNEL].angle = ELBOW_MIN
    kit.servo[SHOULDER_CHANNEL].angle = SHOULDER_MIN

    time.sleep(1)


def cut_off_servos():
    print("Cutting off servo signals...")

    kit.servo[BASE_CHANNEL].angle = None
    kit.servo[ELBOW_CHANNEL].angle = None
    kit.servo[SHOULDER_CHANNEL].angle = None

    print("Servo signals cut off.")


def test_one_motor(name, channel, min_angle, mean_angle, max_angle):
    print(f"\nTesting {name}")

    move_servo(name, channel, mean_angle, 1)
    move_servo(name, channel, max_angle, 1)
    move_servo(name, channel, mean_angle, 1)
    move_servo(name, channel, min_angle, 1)
    move_servo(name, channel, mean_angle, 1)


def test_all_motors_together():
    print("\nTesting ALL motors together at extremes")

    move_all_to_mean()
    move_all_to_max()
    move_all_to_mean()
    move_all_to_min()
    move_all_to_mean()


try:
    print("Starting 3-motor loop test")
    print("Press Ctrl + C to stop safely.")
    print()

    move_all_to_mean()

    while True:
        # Test each motor one by one
        test_one_motor("BASE", BASE_CHANNEL, BASE_MIN, BASE_MEAN, BASE_MAX)
        test_one_motor("ELBOW", ELBOW_CHANNEL, ELBOW_MIN, ELBOW_MEAN, ELBOW_MAX)
        test_one_motor("SHOULDER", SHOULDER_CHANNEL, SHOULDER_MIN, SHOULDER_MEAN, SHOULDER_MAX)

        # Then test all motors together
        test_all_motors_together()

except KeyboardInterrupt:
    print("\nStop requested.")

finally:
    move_all_to_mean()
    time.sleep(1)
    cut_off_servos()
    print("Program ended safely.")
