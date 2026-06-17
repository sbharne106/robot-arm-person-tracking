from flask import Flask, Response, render_template_string
import cv2
import threading
import time
import os


# ==================================================
# FLASK APPLICATION
# ==================================================

app = Flask(__name__)


# ==================================================
# CAMERA SETTINGS
# ==================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

TARGET_FPS = 20
JPEG_QUALITY = 80

MAX_FAILED_FRAMES = 10


# ==================================================
# CAMERA ROTATION
# ==================================================

CAMERA_ROTATION = cv2.ROTATE_90_CLOCKWISE


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

PERSON_CLASS_ID = 15
CONFIDENCE_THRESHOLD = 0.45


# ==================================================
# TRACKING SETTINGS
# ==================================================

DEAD_ZONE_RATIO = 0.12

# Change to True if LEFT and RIGHT are reversed
REVERSE_LEFT_RIGHT = False


# ==================================================
# SHARED VARIABLES
# ==================================================

camera = None
camera_running = False
latest_frame = None

frame_lock = threading.Lock()
camera_lock = threading.Lock()


# ==================================================
# HTML PAGE
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

        html {
            margin: 0;
            padding: 0;

            width: 100%;
            min-height: 100%;

            background-color: #111827;
        }

        body {
            margin: 0;
            padding: 0;

            width: 100%;
            min-height: 100vh;

            background-color: #111827;
            color: white;

            font-family:
                Arial,
                Helvetica,
                sans-serif;

            overflow-x: hidden;
        }

        .container {
            width: 95%;
            max-width: 1000px;
            min-height: 100vh;

            margin: 0 auto;
            padding: 24px 0 20px 0;

            background-color: #111827;

            text-align: center;
        }

        h1 {
            margin: 0 0 6px 0;
        }

        .subtitle {
            margin-bottom: 18px;

            color: #9ca3af;
        }

        .status {
            display: inline-block;

            margin-bottom: 18px;

            padding: 8px 14px;

            color: #4ade80;
            background-color: #16351f;

            border: 1px solid #2f6840;
            border-radius: 20px;
        }

        .camera-container {
            width: 100%;
            max-width: 900px;

            margin: 0 auto;

            background-color: #111827;

            border: 3px solid #374151;
            border-radius: 12px;

            overflow: hidden;

            box-shadow:
                0 10px 35px
                rgba(0, 0, 0, 0.55);
        }

        .camera-feed {
            display: block;

            width: 100%;
            max-width: 100%;
            height: auto;

            margin: 0;
            padding: 0;

            background-color: #111827;

            border: 0;
        }

        .information {
            margin-top: 16px;

            padding-bottom: 20px;

            color: #9ca3af;
            background-color: #111827;

            font-size: 14px;
            line-height: 1.6;
        }

    </style>

</head>

<body>

    <div class="container">

        <h1>
            Robot Arm Person Tracking
        </h1>

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
            Red point: detected person center<br>
            Command: LEFT, RIGHT, STOP or NO PERSON
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
            "Model configuration file not found: "
            f"{PROTOTXT_PATH}"
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model file not found: "
            f"{MODEL_PATH}"
        )

    print("Loading MobileNet SSD model...")

    detection_network = cv2.dnn.readNetFromCaffe(
        PROTOTXT_PATH,
        MODEL_PATH
    )

    print("MobileNet SSD loaded successfully")

    return detection_network


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
                "Could not open camera index "
                f"{CAMERA_INDEX}"
            )

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

        camera.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        actual_width = int(
            camera.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        actual_height = int(
            camera.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        reported_fps = camera.get(
            cv2.CAP_PROP_FPS
        )

    print("Camera opened successfully")

    print(
        "Resolution: "
        f"{actual_width} x {actual_height}"
    )

    print(
        f"Requested FPS: {TARGET_FPS}"
    )

    print(
        "Camera-reported FPS: "
        f"{reported_fps:.2f}"
    )


# ==================================================
# REOPEN CAMERA
# ==================================================

def reopen_camera():

    print("Attempting to reopen camera...")

    try:

        open_camera()

        print("Camera reopened successfully")

        return True

    except Exception as error:

        print(
            "Could not reopen camera: "
            f"{error}"
        )

        time.sleep(1)

        return False


# ==================================================
# DETECT MOST CONFIDENT PERSON
# ==================================================

def detect_person(frame):

    frame_height, frame_width = (
        frame.shape[:2]
    )

    resized_frame = cv2.resize(
        frame,
        (300, 300)
    )

    blob = cv2.dnn.blobFromImage(
        resized_frame,
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
            class_id != PERSON_CLASS_ID
            or confidence < CONFIDENCE_THRESHOLD
            or confidence <= best_confidence
        ):
            continue

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

        start_x = max(
            0,
            start_x
        )

        start_y = max(
            0,
            start_y
        )

        end_x = min(
            frame_width - 1,
            end_x
        )

        end_y = min(
            frame_height - 1,
            end_y
        )

        if (
            end_x <= start_x
            or end_y <= start_y
        ):
            continue

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
        frame_width
        * DEAD_ZONE_RATIO
        / 2
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

    return command


# ==================================================
# PROCESS FRAME
# ==================================================

def process_frame(
    frame,
    measured_fps
):

    frame_height, frame_width = (
        frame.shape[:2]
    )

    person_box, confidence = (
        detect_person(frame)
    )

    tracking_command = "NO PERSON"

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

        tracking_command = (
            get_tracking_command(
                person_center_x,
                frame_width
            )
        )

        # Green person bounding box

        cv2.rectangle(
            frame,
            (start_x, start_y),
            (end_x, end_y),
            (0, 255, 0),
            2
        )

        # Red center point

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

        person_label = (
            "Person: "
            f"{confidence * 100:.1f}%"
        )

        label_y = max(
            100,
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

    # Black information area at top

    cv2.rectangle(
        frame,
        (0, 0),
        (frame_width, 90),
        (0, 0, 0),
        -1
    )

    # FPS text

    cv2.putText(
        frame,
        f"FPS: {measured_fps:.1f}",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    command_color = (
        0,
        255,
        255
    )

    if tracking_command == "STOP":

        command_color = (
            0,
            255,
            0
        )

    elif tracking_command == "NO PERSON":

        command_color = (
            0,
            0,
            255
        )

    cv2.putText(
        frame,
        f"COMMAND: {tracking_command}",
        (15, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        command_color,
        2
    )

    return frame


# ==================================================
# CAMERA CAPTURE LOOP
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
                or frame is None
                or frame.size == 0
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

            # Rotate camera frame

            frame = cv2.rotate(
                frame,
                CAMERA_ROTATION
            )

            # Calculate actual processing FPS

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
                "OpenCV camera error: "
                f"{error}"
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
                "Camera capture error: "
                f"{error}"
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
# MJPEG VIDEO STREAM
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

        success, encoded_frame = (
            cv2.imencode(
                ".jpg",
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    JPEG_QUALITY
                ]
            )
        )

        if not success:

            time.sleep(0.01)
            continue

        frame_bytes = (
            encoded_frame.tobytes()
        )

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

    if (
        not camera_running
        and camera is None
    ):
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
        print()
        print(
            "Open:"
        )
        print(
            "http://<RASPBERRY-PI-IP>:5000"
        )
        print()
        print(
            "Use hostname -I to find "
            "the Raspberry Pi IP address."
        )
        print()
        print(
            "Press Ctrl+C to stop."
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

        print(
            "\nProgram stopped by user"
        )

    except Exception as error:

        print(
            "\nProgram error: "
            f"{error}"
        )

    finally:

        close_camera()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )
