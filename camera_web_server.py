from flask import Flask, Response, render_template_string
import cv2
import threading
import time
import atexit

# --------------------------------------------------
# Flask application
# --------------------------------------------------

app = Flask(__name__)

# --------------------------------------------------
# Camera configuration
# --------------------------------------------------

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

camera = None
latest_frame = None
camera_running = True
frame_lock = threading.Lock()


# --------------------------------------------------
# HTML webpage
# --------------------------------------------------

PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>Robot Arm Camera Feed</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;
            background-color: #111827;
            color: white;
            font-family: Arial, Helvetica, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .container {
            width: 95%;
            max-width: 900px;
            text-align: center;
        }

        h1 {
            margin-bottom: 8px;
        }

        .status {
            color: #4ade80;
            margin-bottom: 20px;
        }

        .camera-container {
            background-color: black;
            border: 3px solid #374151;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }

        .camera-feed {
            display: block;
            width: 100%;
            height: auto;
        }

        .information {
            margin-top: 18px;
            color: #9ca3af;
            font-size: 14px;
        }
    </style>
</head>

<body>

    <div class="container">

        <h1>Robot Arm Camera Feed</h1>

        <div class="status">
            ● Camera server online
        </div>

        <div class="camera-container">
            <img
                class="camera-feed"
                src="/video_feed"
                alt="Raspberry Pi camera feed"
            >
        </div>

        <div class="information">
            Live feed from the Raspberry Pi person-tracking camera
        </div>

    </div>

</body>

</html>
"""


# --------------------------------------------------
# Open the camera
# --------------------------------------------------

def open_camera():
    global camera

    print("Opening camera...")

    # CAP_V4L2 tells OpenCV to use the Linux camera interface
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera at index {CAMERA_INDEX}"
        )

    # Request MJPEG format for better USB/V4L2 performance
    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = camera.get(cv2.CAP_PROP_FPS)

    print("Camera opened successfully")
    print(f"Resolution: {actual_width} x {actual_height}")
    print(f"Requested FPS: {TARGET_FPS}")
    print(f"Camera-reported FPS: {actual_fps}")


# --------------------------------------------------
# Continuously read frames
# --------------------------------------------------

def camera_capture_loop():
    global latest_frame
    global camera_running

    while camera_running:
        success, frame = camera.read()

        if not success:
            print("Warning: Could not read camera frame")
            time.sleep(0.1)
            continue

        # Add a label to the feed
        cv2.putText(
            frame,
            "Robot Arm Camera",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Store the newest frame safely
        with frame_lock:
            latest_frame = frame.copy()


# --------------------------------------------------
# Convert frames into an MJPEG stream
# --------------------------------------------------

def generate_video_stream():
    while True:
        with frame_lock:
            if latest_frame is None:
                frame = None
            else:
                frame = latest_frame.copy()

        if frame is None:
            time.sleep(0.05)
            continue

        # Convert the OpenCV frame into a JPEG image
        success, encoded_frame = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )

        if not success:
            continue

        frame_bytes = encoded_frame.tobytes()

        # Send one JPEG frame as part of the MJPEG stream
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# --------------------------------------------------
# Webpage routes
# --------------------------------------------------

@app.route("/")
def index():
    return render_template_string(PAGE_HTML)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# --------------------------------------------------
# Safely close the camera
# --------------------------------------------------

def close_camera():
    global camera_running

    print("\nClosing camera server...")

    camera_running = False

    if camera is not None:
        camera.release()

    print("Camera released")


atexit.register(close_camera)


# --------------------------------------------------
# Start the program
# --------------------------------------------------

if __name__ == "__main__":
    try:
        open_camera()

        camera_thread = threading.Thread(
            target=camera_capture_loop,
            daemon=True
        )

        camera_thread.start()

        print("\nCamera webpage is running.")
        print("Open the Raspberry Pi IP address on another device.")
        print("Example: http://192.168.1.25:5000")
        print("\nPress Ctrl+C to stop the server.\n")

        app.run(
            host="0.0.0.0",
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False
        )

    except KeyboardInterrupt:
        print("\nProgram stopped by user")

    except Exception as error:
        print(f"Error: {error}")

    finally:
        close_camera()
