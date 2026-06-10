import cv2

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

MODEL = "mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
PROTOTXT = "mobilenet_ssd/MobileNetSSD_deploy.prototxt"

net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera did not open")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera and AI model started")
print("Stand farther back so the camera sees your torso/full body")
print("Press q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read camera frame")
        break

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        0.007843,
        (300, 300),
        127.5
    )

    net.setInput(blob)
    detections = net.forward()

    found_person = False
    best_class = "none"
    best_confidence = 0

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        class_id = int(detections[0, 0, i, 1])

        if class_id < len(CLASSES) and confidence > best_confidence:
            best_confidence = confidence
            best_class = CLASSES[class_id]

        if class_id == 15 and confidence > 0.20:
            found_person = True

            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            startX, startY, endX, endY = box.astype("int")

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"PERSON {confidence * 100:.1f}%",
                (startX, max(20, startY - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    if found_person:
        status = "PERSON DETECTED"
    else:
        status = "NO PERSON"

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
        f"Best guess: {best_class} {best_confidence * 100:.1f}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow("Person AI Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
