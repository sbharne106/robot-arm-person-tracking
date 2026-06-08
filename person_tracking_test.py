
import cv2
import time

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

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Could not open camera")
    exit()

DEAD_ZONE = 15

prev_time = time.time()
last_command = None

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame")
        break

    h, w = frame.shape[:2]
    frame_center_x = w // 2

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        0.007843,
        (300, 300),
        127.5
    )

    net.setInput(blob)
    detections = net.forward()

    command = "NO PERSON"
    best_confidence = 0
    best_box = None

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        if class_id == PERSON_CLASS_ID and confidence > 0.5:
            if confidence > best_confidence:
                best_confidence = confidence
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                best_box = box.astype("int")

    if best_box is not None:
        startX, startY, endX, endY = best_box

        person_center_x = (startX + endX) // 2
        error_x = person_center_x - frame_center_x

        # Direction labels are flipped to match your camera/arm perspective
        if error_x < -DEAD_ZONE:
            command = "RIGHT"
        elif error_x > DEAD_ZONE:
            command = "LEFT"
        else:
            command = "STOP"

        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.circle(frame, (person_center_x, (startY + endY) // 2), 5, (0, 255, 0), -1)

        cv2.putText(
            frame,
            f"person: {best_confidence * 100:.1f}%",
            (startX, max(startY - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    cv2.line(frame, (frame_center_x, 0), (frame_center_x, h), (255, 0, 0), 2)
    cv2.line(frame, (frame_center_x - DEAD_ZONE, 0), (frame_center_x - DEAD_ZONE, h), (255, 255, 0), 1)
    cv2.line(frame, (frame_center_x + DEAD_ZONE, 0), (frame_center_x + DEAD_ZONE, h), (255, 255, 0), 1)

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(frame, f"Command: {command}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if command != last_command:
        print(command)
        last_command = command

    cv2.imshow("Person Tracking Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

