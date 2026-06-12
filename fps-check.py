import cv2
import time

# Open camera
cap = cv2.VideoCapture(0)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Optional: ask camera for 30 FPS
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Could not open camera")
    exit()

prev_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame")
        break

    current_time = time.time()
    time_diff = current_time - prev_time

    if time_diff > 0:
        fps = 1 / time_diff

    prev_time = current_time

    # Show FPS on screen
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("FPS Test", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
