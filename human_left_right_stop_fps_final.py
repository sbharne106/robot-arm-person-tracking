import cv2
import time

# -----------------------------
# MobileNet SSD class labels
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

# -----------------------------
# Load AI model
# -----------------------------
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# -----------------------------
# USB camera setup
# -----------------------------
CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Camera did not open")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# -----------------------------
# Detection / tracking settings
# -----------------------------
CONFIDENCE_THRESHOLD = 0.20

# Reduced dead zone
# Smaller value = more sensitive movement
DEAD_ZONE = 20

# -----------------------------
# FPS averaging variables
# -----------------------------
fps_start_time = time.time()
frame_count = 0
fps = 0

last_command = None

print("Human LEFT / RIGHT / STOP detection started.")
print("Left and right are reversed.")
print("Press q to quit.")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Could not read camera frame")
        break

    h, w = frame.shape[:2]
    frame_center_x = w // 2

    # -----------------------------
    # Averaged FPS calculation
    # -----------------------------
    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - fps_start_time

    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        fps_start_time = current_time

    # -----------------------------
    # Run human detection
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

    # Only keep the best human detection
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        class_id = int(detections[0, 0, i, 1])

        if class_id == PERSON_CLASS_ID and confidence > CONFIDENCE_THRESHOLD:
            if confidence > best_confidence:
                best_confidence = confidence
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                best_box = box.astype("int")

    # -----------------------------
    # Motion command logic
    # -----------------------------
    command = "NO HUMAN"

    if best_box is not None:
        startX, startY, endX, endY = best_box

        # Keep coordinates inside frame
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(w - 1, endX)
        endY = min(h - 1, endY)

        person_center_x = (startX + endX) // 2
        person_center_y = (startY + endY) // 2
        error_x = person_center_x - frame_center_x

        # REVERSED left/right logic
        if abs(error_x) <= DEAD_ZONE:
            command = "STOP"
        elif error_x < 0:
            command = "RIGHT"
        else:
            command = "LEFT"

        # Draw bounding box
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Draw center point of detected human
        cv2.circle(frame, (person_center_x, person_center_y), 6, (0, 255, 255), -1)

        # Show confidence
        label = f"Human {best_confidence * 100:.1f}%"
        cv2.putText(
            frame,
            label,
            (startX, max(25, startY - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Optional debug value
        cv2.putText(
            frame,
            f"error_x: {error_x}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    # Print command only when it changes
    if command != last_command:
        print(command)
        last_command = command

    # -----------------------------
    # Draw guide lines
    # -----------------------------
    left_stop_line = frame_center_x - DEAD_ZONE
    right_stop_line = frame_center_x + DEAD_ZONE

    # Center line
    #cv2.line(frame, (frame_center_x, 0), (frame_center_x, h), (255, 255, 255), 2)

    # Stop zone lines
    #cv2.line(frame, (left_stop_line, 0), (left_stop_line, h), (255, 0, 0), 2)
    #cv2.line(frame, (right_stop_line, 0), (right_stop_line, h), (255, 0, 0), 2)

    # -----------------------------
    # Display command and FPS
    # -----------------------------
    cv2.putText(
        frame,
        f"COMMAND: {command}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("Human LEFT RIGHT STOP with FPS", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
