
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

# MJPG usually gives better FPS for USB webcams
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# Lower camera resolution for higher FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Ask camera for 30 FPS
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Could not open camera")
    exit()

prev_time = time.time()

frame_count = 0
last_detections = None
while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame")
        break

     h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        0.007843,
        (300, 300),
        127.5
        )
    frame_count += 1

    if frame_count % 2 == 0:
    	frame_count += 1

    # Only run AI detection every 2nd frame
    if frame_count % 2 == 0:
    	net.setInput(blob)
    	last_detections = net.forward()

    detections = last_detections

	# If no detection has been run yet, just show the normal frame
    if detections is None:
    	cv2.imshow("Person Detection - SSD MobileNet", frame)

    	if cv2.waitKey(1) & 0xFF == ord("q"):
        	break

        continue

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])

            # Only detect people
            if class_id != PERSON_CLASS_ID:
                continue

            label = "person"

            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            startX, startY, endX, endY = box.astype("int")

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            text = f"{label}: {confidence * 100:.1f}%"
            y = startY - 10 if startY - 10 > 20 else startY + 20

            cv2.putText(
                frame,
                text,
                (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.imshow("Person Detection - SSD MobileNet", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
