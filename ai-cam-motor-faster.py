from adafruit_servokit import ServoKit
import cv2
import time
import os

# ============================================================
# PCA9685 SETUP
# ============================================================

kit = ServoKit(channels=16, address=0x40)

# Change these if your wiring is different
BASE_CHANNEL = 3
ELBOW_CHANNEL = 2
SHOULDER_CHANNEL = 0

# If your servos support 270 degree range
kit.servo[BASE_CHANNEL].actuation_range = 270
kit.servo[ELBOW_CHANNEL].actuation_range = 270
kit.servo[SHOULDER_CHANNEL].actuation_range = 270

# ============================================================
# SERVO CALIBRATION - OFFSET SYSTEM
# ============================================================

# Base calibration
BASE_MIN = 0
BASE_MEAN = 130
BASE_MAX = 270
BASE_DIRECTION = 1

# Elbow calibration
# Offset range:
# raw 100 = +140
# raw 240 = 0
# raw 260 = -20
ELBOW_MIN = 100
ELBOW_MEAN = 240
ELBOW_MAX = 260
ELBOW_DIRECTION = -1

# Shoulder calibration
# Offset range:
# raw 30 = -160
# raw 190 = 0
# raw 270 = +80
SHOULDER_MIN = 30
SHOULDER_MEAN = 190
SHOULDER_MAX = 270
SHOULDER_DIRECTION = 1

# ============================================================
# OFFSET LIMITS
# ============================================================

BASE_OFFSET_MIN = (BASE_MIN - BASE_MEAN) * BASE_DIRECTION
BASE_OFFSET_MAX = (BASE_MAX - BASE_MEAN) * BASE_DIRECTION

SHOULDER_OFFSET_MIN = (SHOULDER_MIN - SHOULDER_MEAN) * SHOULDER_DIRECTION
SHOULDER_OFFSET_MAX = (SHOULDER_MAX - SHOULDER_MEAN) * SHOULDER_DIRECTION

ELBOW_OFFSET_MIN = (ELBOW_MAX - ELBOW_MEAN) * ELBOW_DIRECTION
ELBOW_OFFSET_MAX = (ELBOW_MIN - ELBOW_MEAN) * ELBOW_DIRECTION

BASE_OFFSET_MIN, BASE_OFFSET_MAX = sorted([BASE_OFFSET_MIN, BASE_OFFSET_MAX])
SHOULDER_OFFSET_MIN, SHOULDER_OFFSET_MAX = sorted([SHOULDER_OFFSET_MIN, SHOULDER_OFFSET_MAX])
ELBOW_OFFSET_MIN, ELBOW_OFFSET_MAX = sorted([ELBOW_OFFSET_MIN, ELBOW_OFFSET_MAX])

# ============================================================
# T1 SAFETY CHECK ONLY
# ============================================================

# T1 safety:
# When shoulder is near -160, elbow cannot go below +50.
# So elbow is only allowed from 140 to 50 in that shoulder region.
T1_SHOULDER_LIMIT = -160
T1_ELBOW_MIN_ALLOWED = 50
T1_SHOULDER_BUFFER = 5

# ============================================================
# TRACKING TUNING - FASTER VERSION
# ============================================================

# Deadzones
# Smaller = more sensitive
# Larger = less jitter
X_DEADZONE = 0.08
Y_DEADZONE = 0.04
SIZE_DEADZONE = 0.03

# Movement gains
# Higher = target changes faster when person moves
BASE_GAIN = 5.0
SHOULDER_GAIN = 6.0
ELBOW_GAIN = 8.0

# Max step per frame
# Higher = motor physically moves faster
BASE_MAX_STEP = 5.0
SHOULDER_MAX_STEP = 2.5
ELBOW_MAX_STEP = 4.0

# Direction signs
# If a motor moves opposite, flip its sign.
BASE_TRACK_SIGN = -1
SHOULDER_TRACK_SIGN = -1
ELBOW_TRACK_SIGN = 1

# Target person size in the frame
TARGET_PERSON_HEIGHT_RATIO = 0.55

# Detection confidence threshold
CONFIDENCE_THRESHOLD = 0.45

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Model files
MODEL_DIR = "mobilenet_ssd"
PROTOTXT_PATH = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.prototxt")
MODEL_PATH = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.caffemodel")

# PASCAL VOC class labels used by MobileNet SSD
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

PERSON_CLASS_ID = 15

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def offset_to_angle(offset, mean, direction, min_angle, max_angle):
    """
    Converts offset angle to actual servo raw angle.

    raw_angle = mean + direction * offset
    """
    raw_angle = mean + direction * offset
    return clamp(raw_angle, min_angle, max_angle)


def apply_t1_safety(shoulder_offset, elbow_offset):
    """
    T1 safety:
    If shoulder is near -160, elbow is not allowed below +50.
    """
    if shoulder_offset <= T1_SHOULDER_LIMIT + T1_SHOULDER_BUFFER:
        elbow_offset = clamp(elbow_offset, T1_ELBOW_MIN_ALLOWED, ELBOW_OFFSET_MAX)
    else:
        elbow_offset = clamp(elbow_offset, ELBOW_OFFSET_MIN, ELBOW_OFFSET_MAX)

    return shoulder_offset, elbow_offset


def step_toward(current, target, max_step):
    """
    Moves current value toward target by at most max_step.
    Higher max_step = faster movement.
    """
    diff = target - current

    if abs(diff) <= max_step:
        return target

    if diff > 0:
        return current + max_step
    else:
        return current - max_step


def write_all_motors(base_offset, shoulder_offset, elbow_offset):
    """
    Writes all 3 motors together.
    """

    base_angle = offset_to_angle(
        base_offset,
        BASE_MEAN,
        BASE_DIRECTION,
        BASE_MIN,
        BASE_MAX
    )

    shoulder_angle = offset_to_angle(
        shoulder_offset,
        SHOULDER_MEAN,
        SHOULDER_DIRECTION,
        SHOULDER_MIN,
        SHOULDER_MAX
    )

    elbow_angle = offset_to_angle(
        elbow_offset,
        ELBOW_MEAN,
        ELBOW_DIRECTION,
        ELBOW_MIN,
        ELBOW_MAX
    )

    kit.servo[BASE_CHANNEL].angle = base_angle
    kit.servo[SHOULDER_CHANNEL].angle = shoulder_angle
    kit.servo[ELBOW_CHANNEL].angle = elbow_angle


def disable_all_motors():
    kit.servo[BASE_CHANNEL].angle = None
    kit.servo[SHOULDER_CHANNEL].angle = None
    kit.servo[ELBOW_CHANNEL].angle = None


def go_to_mean_position():
    """
    Smoothly brings all motors back to offset 0.
    Uses the faster max step values too.
    """
    global current_base_offset
    global current_shoulder_offset
    global current_elbow_offset

    target_base = 0
    target_shoulder = 0
    target_elbow = 0

    while True:
        current_base_offset = step_toward(
            current_base_offset,
            target_base,
            BASE_MAX_STEP
        )

        current_shoulder_offset = step_toward(
            current_shoulder_offset,
            target_shoulder,
            SHOULDER_MAX_STEP
        )

        current_elbow_offset = step_toward(
            current_elbow_offset,
            target_elbow,
            ELBOW_MAX_STEP
        )

        current_shoulder_offset, current_elbow_offset = apply_t1_safety(
            current_shoulder_offset,
            current_elbow_offset
        )

        write_all_motors(
            current_base_offset,
            current_shoulder_offset,
            current_elbow_offset
        )

        if (
            current_base_offset == target_base and
            current_shoulder_offset == target_shoulder and
            current_elbow_offset == target_elbow
        ):
            break

        time.sleep(0.02)


def find_largest_person(detections, frame_width, frame_height):
    """
    Finds the largest detected person in the frame.
    Returns bounding box and confidence.
    """

    best_box = None
    best_area = 0
    best_confidence = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        if class_id != PERSON_CLASS_ID:
            continue

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        box = detections[0, 0, i, 3:7] * [
            frame_width,
            frame_height,
            frame_width,
            frame_height
        ]

        x1, y1, x2, y2 = box.astype("int")

        x1 = int(clamp(x1, 0, frame_width - 1))
        y1 = int(clamp(y1, 0, frame_height - 1))
        x2 = int(clamp(x2, 0, frame_width - 1))
        y2 = int(clamp(y2, 0, frame_height - 1))

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width <= 0 or box_height <= 0:
            continue

        area = box_width * box_height

        if area > best_area:
            best_area = area
            best_box = (x1, y1, x2, y2)
            best_confidence = confidence

    return best_box, best_confidence


# ============================================================
# INITIAL MOTOR POSITIONS
# ============================================================

current_base_offset = 0
current_shoulder_offset = 0
current_elbow_offset = 0

target_base_offset = 0
target_shoulder_offset = 0
target_elbow_offset = 0

print("Moving all motors to mean position...")
write_all_motors(0, 0, 0)
time.sleep(1)

# ============================================================
# LOAD AI MODEL
# ============================================================

print("Loading MobileNet SSD model...")

if not os.path.exists(PROTOTXT_PATH):
    raise FileNotFoundError(f"Could not find prototxt file: {PROTOTXT_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Could not find model file: {MODEL_PATH}")

net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

# ============================================================
# CAMERA SETUP
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

print("Camera opened.")
print("Press q to quit. Motors will return to mean and cut off.")

# FPS calculation
fps = 0
frame_counter = 0
fps_start_time = time.time()

# ============================================================
# MAIN LOOP
# ============================================================

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame.")
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        h, w = frame.shape[:2]

        # Create blob for MobileNet SSD
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=0.007843,
            size=(300, 300),
            mean=127.5
        )

        net.setInput(blob)
        detections = net.forward()

        person_box, confidence = find_largest_person(detections, w, h)

        if person_box is not None:
            x1, y1, x2, y2 = person_box

            person_center_x = (x1 + x2) // 2
            person_center_y = (y1 + y2) // 2
            person_height = y2 - y1

            frame_center_x = w // 2
            frame_center_y = h // 2

            # Normalized errors
            # x_error:
            # negative = person left of center
            # positive = person right of center
            x_error = (person_center_x - frame_center_x) / (w / 2)

            # y_error:
            # negative = person above center
            # positive = person below center
            y_error = (person_center_y - frame_center_y) / (h / 2)

            # Person size error
            # positive = person looks too small/far
            # negative = person looks too large/close
            person_height_ratio = person_height / h
            size_error = TARGET_PERSON_HEIGHT_RATIO - person_height_ratio

            # ------------------------------
            # BASE TRACKING
            # ------------------------------
            if abs(x_error) > X_DEADZONE:
                target_base_offset += BASE_TRACK_SIGN * x_error * BASE_GAIN

            # ------------------------------
            # SHOULDER TRACKING
            # ------------------------------
            # Shoulder reacts to both vertical position and person size.
            shoulder_error = y_error + (0.4 * size_error)

            if abs(shoulder_error) > Y_DEADZONE:
                target_shoulder_offset += (
                    SHOULDER_TRACK_SIGN * shoulder_error * SHOULDER_GAIN
                )

            # ------------------------------
            # ELBOW TRACKING
            # ------------------------------
            # Elbow reacts to both person size and vertical position.
            elbow_error = size_error + (0.5 * y_error)

            if abs(elbow_error) > SIZE_DEADZONE:
                target_elbow_offset += ELBOW_TRACK_SIGN * elbow_error * ELBOW_GAIN

            # Clamp target offsets to mechanical limits
            target_base_offset = clamp(
                target_base_offset,
                BASE_OFFSET_MIN,
                BASE_OFFSET_MAX
            )

            target_shoulder_offset = clamp(
                target_shoulder_offset,
                SHOULDER_OFFSET_MIN,
                SHOULDER_OFFSET_MAX
            )

            target_elbow_offset = clamp(
                target_elbow_offset,
                ELBOW_OFFSET_MIN,
                ELBOW_OFFSET_MAX
            )

            # Apply T1 safety to target positions
            target_shoulder_offset, target_elbow_offset = apply_t1_safety(
                target_shoulder_offset,
                target_elbow_offset
            )

            # Draw person box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"Person: {confidence * 100:.1f}%"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Draw person center point
            cv2.circle(frame, (person_center_x, person_center_y), 5, (0, 0, 255), -1)

            # Draw frame center
            cv2.circle(frame, (frame_center_x, frame_center_y), 5, (255, 0, 0), -1)

            # Debug tracking errors
            cv2.putText(
                frame,
                f"x:{x_error:.2f} y:{y_error:.2f} size:{size_error:.2f}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                f"shoulder_err:{shoulder_error:.2f} elbow_err:{elbow_error:.2f}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

        else:
            cv2.putText(
                frame,
                "No person detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # ====================================================
        # COORDINATED MOTOR MOVEMENT
        # All 3 motors move together toward their targets.
        # Faster max step values make the motors respond quicker.
        # ====================================================

        current_base_offset = step_toward(
            current_base_offset,
            target_base_offset,
            BASE_MAX_STEP
        )

        current_shoulder_offset = step_toward(
            current_shoulder_offset,
            target_shoulder_offset,
            SHOULDER_MAX_STEP
        )

        current_elbow_offset = step_toward(
            current_elbow_offset,
            target_elbow_offset,
            ELBOW_MAX_STEP
        )

        # Apply T1 safety again to actual current movement
        current_shoulder_offset, current_elbow_offset = apply_t1_safety(
            current_shoulder_offset,
            current_elbow_offset
        )

        write_all_motors(
            current_base_offset,
            current_shoulder_offset,
            current_elbow_offset
        )

        # ====================================================
        # FPS CALCULATION
        # ====================================================

        frame_counter += 1
        elapsed_time = time.time() - fps_start_time

        if elapsed_time >= 1.0:
            fps = frame_counter / elapsed_time
            frame_counter = 0
            fps_start_time = time.time()

        # ====================================================
        # DISPLAY DEBUG INFO
        # ====================================================

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, h - 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Base offset: {current_base_offset:.1f}",
            (20, h - 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Shoulder offset: {current_shoulder_offset:.1f}",
            (20, h - 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Elbow offset: {current_elbow_offset:.1f}",
            (20, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Targets B/S/E: {target_base_offset:.1f}, {target_shoulder_offset:.1f}, {target_elbow_offset:.1f}",
            (20, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # Safety message
        if current_shoulder_offset <= T1_SHOULDER_LIMIT + T1_SHOULDER_BUFFER:
            cv2.putText(
                frame,
                "T1 SAFETY ACTIVE",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        cv2.imshow("AI 3 Motor Person Tracking", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("q pressed. Returning motors to mean position...")
            break

except KeyboardInterrupt:
    print("Keyboard interrupt. Returning motors to mean position...")

finally:
    cap.release()
    cv2.destroyAllWindows()

    go_to_mean_position()

    time.sleep(0.5)
    disable_all_motors()

    print("Motors returned to mean and cut off.")
