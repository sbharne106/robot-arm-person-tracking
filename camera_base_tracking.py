import cv2
import time
from adafruit_servokit import ServoKit

# -----------------------------
# PCA9685 / SERVO SETUP
# -----------------------------
kit = ServoKit(channels=16)

# Change this if your base motor is on a different channel
BASE_CHANNEL = 3

# Base offset system values
BASE_MEAN = 90
BASE_MIN = 0
BASE_MAX = 180

# Start base at mean
base_angle = BASE_MEAN
kit.servo[BASE_CHANNEL].angle = base_angle
time.sleep(1)

# -----------------------------
# CAMERA SETUP
# -----------------------------
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

FRAME_WIDTH = 640
FRAME_CENTER_X = FRAME_WIDTH // 2

# Smaller dead zone = more sensitive tracking
DEAD_ZONE = 40

# How much the servo moves each time
STEP_SIZE = 2

# -----------------------------
# AI MODEL SETUP
# -----------------------------
prototxt_path = "mobilenet_ssd/MobileNetSSD_deploy.prototxt"
model_path = "mobilenet_ssd/MobileNetSSD_deploy.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

PERSON_CLASS_ID = 15
CONFIDENCE_THRESHOLD = 0.5

# -----------------------------
# FPS SETUP
# -----------------------------
prev_time = time.time()
fps = 0

print("Camera + base tracking started.")
print("Press q to quit.")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame from camera.")
            break

        h, w = frame.shape[:2]

        # AI input
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843,
            (300, 300),
            127.5
        )

        net.setInput(blob)
        detections = net.forward()

        person_found = False
        best_confidence = 0
        best_box = None

        # Find the strongest person detection
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])

            if class_id == PERSON_CLASS_ID and confidence > CONFIDENCE_THRESHOLD:
                if confidence > best_confidence:
                    best_confidence = confidence

                    box = detections[0, 0, i, 3:7] * [w, h, w, h]
                    best_box = box.astype("int")
                    person_found = True

        command = "STOP"

        if person_found:
            startX, startY, endX, endY = best_box

            person_center_x = (startX + endX) // 2
            error_x = person_center_x - FRAME_CENTER_X

            # Draw person box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.circle(frame, (person_center_x, (startY + endY) // 2), 5, (0, 0, 255), -1)

            # Tracking logic
            # If person is to the right, rotate base right
            # If person is to the left, rotate base left
            if error_x > DEAD_ZONE:
                command = "RIGHT"
                base_angle -= STEP_SIZE

            elif error_x < -DEAD_ZONE:
                command = "LEFT"
                base_angle += STEP_SIZE

            else:
                command = "CENTERED"

            # Keep angle inside safe range
            base_angle = max(BASE_MIN, min(BASE_MAX, base_angle))

            # Move base servo
            kit.servo[BASE_CHANNEL].angle = base_angle

            cv2.putText(
                frame,
                f"Person: {best_confidence:.2f}",
                (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        else:
            command = "NO PERSON"

        # Calculate FPS
        current_time = time.time()
        dt = current_time - prev_time

        if dt > 0:
            fps = 1 / dt

        prev_time = current_time

        # Display information
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Command: {command}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Base Angle: {base_angle}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw center reference line
        cv2.line(frame, (FRAME_CENTER_X, 0), (FRAME_CENTER_X, h), (255, 0, 0), 2)

        cv2.imshow("Camera Base Tracking", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Quitting. Returning base to mean position...")
            break

except KeyboardInterrupt:
    print("Program stopped manually. Returning base to mean position...")

finally:
    # Return base to mean position before shutting off
    kit.servo[BASE_CHANNEL].angle = BASE_MEAN
    time.sleep(1)

    # Cut off servo signal
    kit.servo[BASE_CHANNEL].angle = None

    cap.release()
    cv2.destroyAllWindows()

    print("Base returned to mean and servo signal cut off.")
