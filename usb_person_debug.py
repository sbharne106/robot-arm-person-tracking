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

if not cap.isOpened():
    print("Could not open USB camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

CONFIDENCE_THRESHOLD = 0.20

prev_time = time.time()
fps = 0

print("Debug person detector started.")
print("Stand farther back so the camera can see your upper body/full body.")
print("Press q to quit.")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Could not read frame")
        break

    h, w = frame.shape[:2]

    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        0.007843,
        (300, 300),
        127.5
    )

    net.setInput(blob)
    detections = net.forward()

    best_label = "nothing"
    best_conf = 0

    person_detected = False

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        class_id = int(detections[0, 0, i, 1])

        if confidence > best_conf and class_id < len(CLASSES):
            best_conf = confidence
            best_label = CLASSES[class_id]

        if confidence > CONFIDENCE_THRESHOLD and class_id < len(CLASSES):
            label_name = CLASSES[class_id]

            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            startX, startY, endX, endY = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)

            if class_id == PERSON_CLASS_ID:
                person_detected = True
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            text = f"{label_name}: {confidence * 100:.1f}%"
            cv2.putText(
                frame,
                text,
                (startX, max(20, startY - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    status = "PERSON DETECTED" if person_detected else "NO PERSON"

    cv2.putText(
        frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Best: {best_label} {best_conf * 100:.1f}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow("Person Detection Debug", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
