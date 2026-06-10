import cv2
import time

# -----------------------------
# MobileNet SSD setup
# -----------------------------

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

PERSON_CLASS_ID = 15

MODEL = "mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
PROTOTXT = "mobilenet_ssd/MobileNetSSD_deploy.prototxt"

net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# -----------------------------
# Camera setup
# -----------------------------

# USB camera is usually 0.
# If it does not work, change this to 1 or 2.
CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Could not open USB camera")
    exit()

# Keep this lower for faster person tracking
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# -----------------------------
# Tracking settings
# -----------------------------

CONFIDENCE_THRESHOLD = 0.45

# Dead zone means how close to the center the person must be
# before we say STOP.
# Bigger number = less sensitive.
DEAD_ZONE = 60

last_command = None
prev_time = time.time()
fps = 0

print("USB person tracking started.")
print("Press q in the camera window to quit.")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Could not read frame from camera")
        break

    h, w = frame.shape[:2]
    frame_center_x = w // 2

    # -----------------------------
    # Calculate FPS
    # -----------------------------

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # -----------------------------
    # Person detection
    # -----------------------------

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        0.007843,
        (300, 300),
        127.5
    )

    net.setInput(blob)
    detections = net.forward()

    best_confidence = 0
    best_box = None

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        if class_id == PERSON_CLASS_ID and confidence > CONFIDENCE_THRESHOLD:
            if confidence > best_confidence:
                best_confidence = confidence

                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                best_box = box.astype("int")

    # Default command if no person is detected
    command = "NO PERSON"

    if best_box is not None:
        startX, startY, endX, endY = best_box

        # Keep box within frame limits
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(w - 1, endX)
        endY = min(h - 1, endY)

        person_center_x = (startX + endX) // 2
        error_x = person_center_x - frame_center_x

        # -----------------------------
        # LEFT / RIGHT / STOP logic
        # -----------------------------
        # If person is left of center, command LEFT.
        # If person is right of center, command RIGHT.
        # If person is close to center, command STOP.

        if abs(error_x) <= DEAD_ZONE:
            command = "STOP"
        elif error_x < 0:
            command = "LEFT"
        else:
            command = "RIGHT"

        # Draw person box
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        label = f"person: {best_confidence * 100:.1f}%"
        cv2.putText(
            frame,
            label,
            (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # Draw person center
        cv2.circle(frame, (person_center_x, h // 2), 6, (0, 255, 255), -1)

    # -----------------------------
    # Print command only when it changes
    # -----------------------------

    if command != last_command:
        print(command)
        last_command = command

    # -----------------------------
    # Draw guide lines
    # -----------------------------

    left_stop_line = frame_center_x - DEAD_ZONE
    right_stop_line = frame_center_x + DEAD_ZONE

    # Center line
    cv2.line(frame, (frame_center_x, 0), (frame_center_x, h), (255, 255, 255), 2)

    # Stop zone lines
    cv2.line(frame, (left_stop_line, 0), (left_stop_line, h), (255, 0, 0), 2)
    cv2.line(frame, (right_stop_line, 0), (right_stop_line, h), (255, 0, 0), 2)

    # Display command
    cv2.putText(
        frame,
        f"COMMAND: {command}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # Display FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("USB Person Tracking - LEFT RIGHT STOP", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
