
import cv2 
import time 
cap =cv2.VideoCapture(0)

if not cap.isOpened():
	print("Could not open camera")
	exit()
prev_time = time.time()
while True:
	ret, frame = cap.read()

	if not ret: 
		print("Could not read frame")
		break 

	#calculate FPS 
	current_time = time.time()
	fps = 1/(current_time-prev_time)
	prev_time = current_time

	#put fps text on the frame 
	cv2.putText(frame, f"FPS: {fps:.2f}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 2)
 
	cv2.imshow("USB Camera Feed", frame)

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break 
cap.release
cv2.destroyAllWindows()
