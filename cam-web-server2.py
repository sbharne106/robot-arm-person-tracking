from flask import Flask, Response, render_template_string
import cv2
import threading
import time
import os


# ==================================================
# FLASK WEB APPLICATION
# ==================================================

app = Flask(__name__)


# ==================================================
# CAMERA SETTINGS
# ==================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Lower than the camera's maximum for more stable streaming
TARGET_FPS = 20

# JPEG quality used for browser streaming
JPEG_QUALITY = 80

# Reopen camera after this many consecutive failures
MAX_FAILED_FRAMES = 10


# ==================================================
# PERSON DETECTION SETTINGS
# ==================================================

MODEL_DIR = "mobilenet_ssd"

PROTOTXT_PATH = os.path.join(
    MODEL_DIR,
    "MobileNetSSD_deploy.prototxt"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "MobileNetSSD_deploy.caffemodel"
)

# MobileNet SSD class number for a person
PERSON_CLASS_ID = 15

CONFIDENCE_THRESHOLD = 0.45


# ==================================================
# TRACKING SETTINGS
# ==================================================

# Width of the center dead zone as a fraction
# of the complete frame width
DEAD_ZONE_RATIO = 0.12

# Change this to True if left and right are reversed
REVERSE_LEFT_RIGHT = False


# ==================================================
# SHARED PROGRAM VARIABLES
# ==================================================

camera = None
camera_running = False

latest_frame = None

frame_lock = threading.Lock()
camera_lock = threading.Lock()


# ==================================================
# HTML WEBPAGE
# ==================================================

PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>Robot Arm Person Tracking</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;
            background: #111827;
            color: white;
            font-family: Arial, Helvetica, sans-serif;
        }

        .container {
            width: 95%;
            max-width: 1000px;
            margin: auto;
            padding: 25px 0;
            text-align: center;
        }

        h1 {
            margin: 0 0 8px 0;
        }

        .subtitle {
            color: #9ca3af;
            margin-bottom: 18px;
        }

        .status {
            display: inline-block;
            background: #16351f;
            color: #4ade80;
            border: 1px solid #2f6840;
            padding: 8px 14px;
            border-radius: 20px;
            margin-bottom: 18px;
        }

        .camera-container {
            width: 100%;
            background: black;
            border: 3px solid #374151;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 10px 35px rgba(0, 0, 0, 0.55);
        }

        .camera-feed {
            display: block;
            width: 100%;
            height: auto;
        }

        .information {
            color: #9ca3af;
            margin-top: 18px;
            line-height: 1.6;
        }

        .warning {
            color: #fbbf24;
            margin-top: 12px;
            font-size: 14px;
        }
    </style>
</head>

<body>

    <div class="container">

        <h1>Robot Arm Person Tracking</h1>

        <div class="subtitle">
            Raspberry Pi live camera and AI detection feed
        </div>

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
            Green box: detected person<br>
            Blue lines: center tracking dead zone<br>
            Command: LEFT, RIGHT, STOP or NO PERSON
        </div>

        <div class="warning">
            This local webpage does not currently use a password.
        </div>

    </div>

</body>

</html>
"""


# ==================================================
# LOAD MOBILENET SSD MODEL
# ==================================================

def load_detection_model():
    if not os.path.exists(PROTOTXT_PATH):
        raise FileNotFoundError(
            f"Model configuration file not found: "
            f"{PROTOTXT_PATH}"
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    print("Loading MobileNet SSD model...")

    network = cv2.dnn.readNetFromCaffe(
        PROTOTXT_PATH,
        MODEL_PATH
    )

    print("MobileNet SSD loaded successfully")

    return network


# Load the model once when the program starts
net = load_detection_model()


# ==================================================
# OPEN CAMERA
# ==================================================

def open_camera():
    global camera

    print("Opening camera...")

    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None

        camera = cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_V4L2
        )

        if not camera.isOpened():
            raise RuntimeError(
                f"Could not open camera index "
                f"{CAMERA_INDEX}"
            )

        # Do not force MJPEG because this caused
        # the empty buffer imdecode error.
        camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            FRAME_WIDTH
        )

        camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            FRAME_HEIGHT
        )

        camera.set(
            cv2.CAP_PROP_FPS,
            TARGET_FPS
        )

        # Keep only the newest camera frame when possible
        camera.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        actual_width = int(
            camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        actual_height = int(
            camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        reported_fps = camera.get(
            cv2.CAP_PROP_FPS
        )

    print("Camera opened successfully")
    print(
        f"Resolution: "
        f"{actual_width} x {actual_height}"
    )
    print(f"Requested FPS: {TARGET_FPS}")
    print(
        f"Camera-reported FPS: "
        f"{reported_fps:.2f}"
    )


# ==================================================
# SAFELY REOPEN CAMERA
# ==================================================

def reopen_camera():
    print("Attempting to reopen camera...")

    try:
        with camera_lock:
            if camera is not None:
                camera.release()

        time.sleep(1)

        open_camera()

        print("Camera reopened successfully")
        return True

    except Exception as error:
        print(f"Could not reopen camera: {error}")
        time.sleep(1)
        return False


# ==================================================
# DETECT THE MOST CONFIDENT PERSON
# ==================================================

def detect_person(frame):
    frame_height, frame_width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        scalefactor=0.007843,
        size=(300, 300),
        mean=127.5
    )

    net.setInput(blob)
    detections = net.forward()

    best_person = None
    best_confidence = 0.0

    for detection_index in range(
        detections.shape[2]
    ):
        confidence = float(
            detections[
                0,
                0,
                detection_index,
                2
            ]
        )

        class_id = int(
            detections[
                0,
                0,
                detection_index,
                1
            ]
        )

        if (
            class_id == PERSON_CLASS_ID
            and
            confidence >= CONFIDENCE_THRESHOLD
            and
            confidence > best_confidence
        ):
            box = detections[
                0,
                0,
                detection_index,
                3:7
            ]

            box = box * [
                frame_width,
                frame_height,
                frame_width,
                frame_height
            ]

            start_x, start_y, end_x, end_y = (
                box.astype(int)
            )

            # Prevent bounding box coordinates
            # from going outside the frame
            start_x = max(0, start_x)
            start_y = max(0, start_y)

            end_x = min(
                frame_width - 1,
                end_x
            )

            end_y = min(
                frame_height - 1,
                end_y
            )

            if (
                end_x > start_x
                and
                end_y > start_y
            ):
                best_person = (
                    start_x,
                    start_y,
                    end_x,
                    end_y
                )

                best_confidence = confidence

    return best_person, best_confidence


# ==================================================
# CALCULATE TRACKING COMMAND
# ==================================================

def get_tracking_command(
    person_center_x,
    frame_width
):
    frame_center_x = frame_width // 2

    dead_zone_half_width = int(
        frame_width * DEAD_ZONE_RATIO / 2
    )

    left_boundary = (
        frame_center_x
        - dead_zone_half_width
    )

    right_boundary = (
        frame_center_x
        + dead_zone_half_width
    )

    if person_center_x < left_boundary:
        command = "LEFT"

    elif person_center_x > right_boundary:
        command = "RIGHT"

    else:
        command = "STOP"

    if REVERSE_LEFT_RIGHT:
        if command == "LEFT":
            command = "RIGHT"

        elif command == "RIGHT":
            command = "LEFT"

    return (
        command,
        left_boundary,
        right_boundary
    )


# ==================================================
# DRAW PERSON TRACKING INFORMATION
# ==================================================

def process_frame(frame, measured_fps):
    frame_height, frame_width = frame.shape[:2]

    person_box, confidence = detect_person(frame)

    frame_center_x = frame_width // 2

    dead_zone_half_width = int(
        frame_width * DEAD_ZONE_RATIO / 2
    )

    left_boundary = (
        frame_center_x
        - dead_zone_half_width
    )

    right_boundary = (
        frame_center_x
        + dead_zone_half_width
    )

    tracking_command = "NO PERSON"

    # Draw dead-zone lines
    cv2.line(
        frame,
        (left_boundary, 0),
        (left_boundary, frame_height),
        (255, 150, 0),
        2
    )

    cv2.line(
        frame,
        (right_boundary, 0),
        (right_boundary, frame_height),
        (255, 150, 0),
        2
    )

    # Draw frame center
    cv2.line(
        frame,
        (frame_center_x, 0),
        (frame_center_x, frame_height),
        (150, 150, 150),
        1
    )

    if person_box is not None:
        (
            start_x,
            start_y,
            end_x,
            end_y
        ) = person_box

        person_center_x = (
            start_x + end_x
        ) // 2

        person_center_y = (
            start_y + end_y
        ) // 2

        (
            tracking_command,
            left_boundary,
            right_boundary
        ) = get_tracking_command(
            person_center_x,
            frame_width
        )

        # Person bounding box
        cv2.rectangle(
            frame,
            (start_x, start_y),
            (end_x, end_y),
            (0, 255, 0),
            2
        )

        # Person center point
        cv2.circle(
            frame,
            (
                person_center_x,
                person_center_y
            ),
            6,
            (0, 0, 255),
            -1
        )

        # Draw a line from frame center
        # to detected person center
        cv2.line(
            frame,
            (
                frame_center_x,
                frame_height // 2
            ),
            (
                person_center_x,
                person_center_y
            ),
            (0, 255, 255),
            2
        )

        person_label = (
            f"Person: "
            f"{confidence * 100:.1f}%"
        )

        label_y = max(
            25,
            start_y - 10
        )

        cv2.putText(
            frame,
            person_label,
            (start_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

    # Draw a black information background
    cv2.rectangle(
        frame,
        (0, 0),
        (frame_width, 82),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"FPS: {measured_fps:.1f}",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    command_color = (0, 255, 255)

    if tracking_command == "STOP":
        command_color = (0, 255, 0)

    elif tracking_command == "NO PERSON":
        command_color = (0, 0, 255)

    cv2.putText(
        frame,
        f"COMMAND: {tracking_command}",
        (15, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        command_color,
        2
    )

    return frame


# ==================================================
# CAMERA CAPTURE THREAD
# ==================================================

def camera_capture_loop():
    global latest_frame
    global camera_running

    failed_frames = 0

    fps_frame_count = 0
    measured_fps = 0.0
    fps_start_time = time.monotonic()

    while camera_running:
        try:
            with camera_lock:
                if camera is None:
                    success = False
                    frame = None
                else:
                    success, frame = camera.read()

            if (
                not success
                or
                frame is None
                or
                frame.size == 0
            ):
                failed_frames += 1

                print(
                    "Warning: Invalid camera frame "
                    f"({failed_frames}/"
                    f"{MAX_FAILED_FRAMES})"
                )

                time.sleep(0.1)

                if (
                    failed_frames
                    >= MAX_FAILED_FRAMES
                ):
                    reopen_camera()
                    failed_frames = 0

                continue

            failed_frames = 0

            # Calculate the actual processed FPS
            fps_frame_count += 1

            current_time = time.monotonic()

            elapsed_time = (
                current_time
                - fps_start_time
            )

            if elapsed_time >= 1.0:
                measured_fps = (
                    fps_frame_count
                    / elapsed_time
                )

                fps_frame_count = 0
                fps_start_time = current_time

            processed_frame = process_frame(
                frame,
                measured_fps
            )

            with frame_lock:
                latest_frame = (
                    processed_frame.copy()
                )

        except cv2.error as error:
            print(
                f"OpenCV camera error: {error}"
            )

            failed_frames += 1
            time.sleep(0.25)

            if (
                failed_frames
                >= MAX_FAILED_FRAMES
            ):
                reopen_camera()
                failed_frames = 0

        except Exception as error:
            print(
                f"Camera capture error: {error}"
            )

            failed_frames += 1
            time.sleep(0.25)

            if (
                failed_frames
                >= MAX_FAILED_FRAMES
            ):
                reopen_camera()
                failed_frames = 0


# ==================================================
# GENERATE MJPEG STREAM
# ==================================================

def generate_video_stream():
    while camera_running:
        with frame_lock:
            if latest_frame is None:
                frame = None
            else:
                frame = latest_frame.copy()

        if frame is None:
            time.sleep(0.05)
            continue

        success, encoded_frame = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                JPEG_QUALITY
            ]
        )

        if not success:
            time.sleep(0.01)
            continue

        frame_bytes = encoded_frame.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# ==================================================
# FLASK ROUTES
# ==================================================

@app.route("/")
def index():
    return render_template_string(
        PAGE_HTML
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_video_stream(),
        mimetype=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        )
    )


@app.route("/health")
def health():
    if latest_frame is None:
        return {
            "status": "starting",
            "camera": "waiting for frame"
        }, 503

    return {
        "status": "online",
        "camera": "working"
    }


# ==================================================
# CLOSE CAMERA
# ==================================================

def close_camera():
    global camera_running
    global camera

    if not camera_running and camera is None:
        return

    print("\nClosing camera server...")

    camera_running = False

    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None

    print("Camera released")


# ==================================================
# START PROGRAM
# ==================================================

if __name__ == "__main__":
    try:
        open_camera()

        camera_running = True

        camera_thread = threading.Thread(
            target=camera_capture_loop,
            daemon=True
        )

        camera_thread.start()

        print()
        print("Camera webpage is running.")
        print(
            "Open the following address on "
            "another device:"
        )
        print(
            "http://<RASPBERRY-PI-IP>:5000"
        )
        print()
        print(
            "Use 'hostname -I' to find "
            "the Raspberry Pi IP address."
        )
        print()
        print(
            "Press Ctrl+C to stop the server."
        )
        print()

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
        print(f"\nProgram error: {error}")

    finally:
        close_camera()
