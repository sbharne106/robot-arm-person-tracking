#!/usr/bin/env python3

from adafruit_servokit import ServoKit
import cv2
import time
import os
import signal
import socket
import threading

from flask import Flask, Response, jsonify, render_template_string
from werkzeug.serving import make_server


# ============================================================
# PCA9685 SETUP
# ============================================================

kit = ServoKit(channels=16, address=0x40)

BASE_CHANNEL = 3
ELBOW_CHANNEL = 2
SHOULDER_CHANNEL = 0

kit.servo[BASE_CHANNEL].actuation_range = 270
kit.servo[ELBOW_CHANNEL].actuation_range = 270
kit.servo[SHOULDER_CHANNEL].actuation_range = 270


# ============================================================
# SERVO CALIBRATION - OFFSET SYSTEM
# ============================================================

BASE_MIN = 0
BASE_MEAN = 130
BASE_MAX = 270
BASE_DIRECTION = 1

ELBOW_MIN = 100
ELBOW_MEAN = 240
ELBOW_MAX = 260
ELBOW_DIRECTION = -1

SHOULDER_MIN = 30
SHOULDER_MEAN = 190
SHOULDER_MAX = 270
SHOULDER_DIRECTION = 1


# ============================================================
# OFFSET LIMITS
# ============================================================

BASE_OFFSET_MIN = (BASE_MIN - BASE_MEAN) * BASE_DIRECTION
BASE_OFFSET_MAX = (BASE_MAX - BASE_MEAN) * BASE_DIRECTION

SHOULDER_OFFSET_MIN = (SHOULDER_MIN - SHOULDER_MEAN) * SHOULDER_DIRECTION
SHOULDER_OFFSET_MAX = (SHOULDER_MAX - SHOULDER_MEAN) * SHOULDER_DIRECTION

ELBOW_OFFSET_MIN = (ELBOW_MAX - ELBOW_MEAN) * ELBOW_DIRECTION
ELBOW_OFFSET_MAX = (ELBOW_MIN - ELBOW_MEAN) * ELBOW_DIRECTION

BASE_OFFSET_MIN, BASE_OFFSET_MAX = sorted([BASE_OFFSET_MIN, BASE_OFFSET_MAX])
SHOULDER_OFFSET_MIN, SHOULDER_OFFSET_MAX = sorted([SHOULDER_OFFSET_MIN, SHOULDER_OFFSET_MAX])
ELBOW_OFFSET_MIN, ELBOW_OFFSET_MAX = sorted([ELBOW_OFFSET_MIN, ELBOW_OFFSET_MAX])


# ============================================================
# T1 SAFETY CHECK ONLY
# ============================================================

T1_SHOULDER_LIMIT = -160
T1_ELBOW_MIN_ALLOWED = 50
T1_SHOULDER_BUFFER = 5


# ============================================================
# FAST TRACKING TUNING
# ============================================================

X_DEADZONE = 0.08
Y_DEADZONE = 0.04
SIZE_DEADZONE = 0.03

BASE_GAIN = 5.0
SHOULDER_GAIN = 6.0
ELBOW_GAIN = 8.0

BASE_MAX_STEP = 15.0
SHOULDER_MAX_STEP = 10.5
ELBOW_MAX_STEP = 10.5

BASE_TRACK_SIGN = -1
SHOULDER_TRACK_SIGN = -1
ELBOW_TRACK_SIGN = 1

TARGET_PERSON_HEIGHT_RATIO = 0.55
CONFIDENCE_THRESHOLD = 0.45


# ============================================================
# CAMERA SETTINGS
# ============================================================

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30

# Change to cv2.ROTATE_90_COUNTERCLOCKWISE if needed.
CAMERA_ROTATION = cv2.ROTATE_90_CLOCKWISE

JPEG_QUALITY = 80


# ============================================================
# MODEL FILES
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "mobilenet_ssd")
PROTOTXT_PATH = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.prototxt")
MODEL_PATH = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.caffemodel")

PERSON_CLASS_ID = 15


# ============================================================
# WEB SERVER
# ============================================================

WEB_HOST = "0.0.0.0"
WEB_PORT = 5000

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Robot Arm Person Tracking</title>
    <style>
        * { box-sizing: border-box; }

        html, body {
            margin: 0;
            padding: 0;
            min-height: 100%;
            background: #111827;
            color: white;
            font-family: Arial, Helvetica, sans-serif;
        }

        .container {
            width: 95%;
            max-width: 1050px;
            margin: 0 auto;
            padding: 22px 0 30px;
            text-align: center;
        }

        h1 { margin: 0 0 6px; }

        .subtitle {
            color: #9ca3af;
            margin-bottom: 18px;
        }

        .video-wrapper {
            width: 100%;
            overflow: hidden;
            border: 2px solid #374151;
            border-radius: 12px;
            background: black;
            line-height: 0;
        }

        #camera-feed {
            display: block;
            width: 100%;
            height: auto;
            margin: 0;
            padding: 0;
            background: black;
        }

        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
            gap: 10px;
            margin-top: 16px;
        }

        .status-card {
            background: #1f2937;
            border: 1px solid #374151;
            border-radius: 10px;
            padding: 13px 10px;
        }

        .label {
            color: #9ca3af;
            font-size: 13px;
            margin-bottom: 5px;
        }

        .value {
            font-size: 17px;
            font-weight: bold;
        }

        .online { color: #4ade80; }
        .offline { color: #f87171; }

        .connection {
            margin-top: 14px;
            color: #9ca3af;
            font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Robot Arm Person Tracking</h1>
        <div class="subtitle">Live camera, AI detection and motor tracking</div>

        <div class="video-wrapper">
            <img id="camera-feed" src="/video_feed" alt="Live robot arm camera feed">
        </div>

        <div class="status-grid">
            <div class="status-card">
                <div class="label">System</div>
                <div class="value" id="system">Connecting</div>
            </div>

            <div class="status-card">
                <div class="label">Person</div>
                <div class="value" id="person">--</div>
            </div>

            <div class="status-card">
                <div class="label">Confidence</div>
                <div class="value" id="confidence">--</div>
            </div>

            <div class="status-card">
                <div class="label">FPS</div>
                <div class="value" id="fps">--</div>
            </div>

            <div class="status-card">
                <div class="label">Base Offset</div>
                <div class="value" id="base">--</div>
            </div>

            <div class="status-card">
                <div class="label">Shoulder Offset</div>
                <div class="value" id="shoulder">--</div>
            </div>

            <div class="status-card">
                <div class="label">Elbow Offset</div>
                <div class="value" id="elbow">--</div>
            </div>

            <div class="status-card">
                <div class="label">T1 Safety</div>
                <div class="value" id="t1">--</div>
            </div>
        </div>

        <div class="connection" id="connection">Connecting to Raspberry Pi...</div>
    </div>

    <script>
        async function updateStatus() {
            const connection = document.getElementById("connection");

            try {
                const response = await fetch("/status", { cache: "no-store" });
                if (!response.ok) throw new Error("Status request failed");

                const data = await response.json();

                document.getElementById("system").textContent =
                    data.camera_running ? "ONLINE" : "OFFLINE";

                document.getElementById("system").className =
                    data.camera_running ? "value online" : "value offline";

                document.getElementById("person").textContent =
                    data.person_detected ? "DETECTED" : "NOT DETECTED";

                document.getElementById("confidence").textContent =
                    data.confidence.toFixed(1) + "%";

                document.getElementById("fps").textContent =
                    data.fps.toFixed(1);

                document.getElementById("base").textContent =
                    data.base_offset.toFixed(1);

                document.getElementById("shoulder").textContent =
                    data.shoulder_offset.toFixed(1);

                document.getElementById("elbow").textContent =
                    data.elbow_offset.toFixed(1);

                document.getElementById("t1").textContent =
                    data.t1_active ? "ACTIVE" : "INACTIVE";

                connection.textContent = "Connected to " + data.hostname;
                connection.className = "connection online";
            } catch (error) {
                document.getElementById("system").textContent = "OFFLINE";
                document.getElementById("system").className = "value offline";
                connection.textContent = "Unable to reach Raspberry Pi";
                connection.className = "connection offline";
            }
        }

        updateStatus();
        setInterval(updateStatus, 1000);
    </script>
</body>
</html>
"""


# ============================================================
# SHARED STATE
# ============================================================

frame_lock = threading.Lock()
state_lock = threading.Lock()
motor_lock = threading.Lock()
shutdown_event = threading.Event()

latest_jpeg = None
camera_running = False
person_detected = False
current_confidence = 0.0
fps = 0.0

current_base_offset = 0
current_shoulder_offset = 0
current_elbow_offset = 0

target_base_offset = 0
target_shoulder_offset = 0
target_elbow_offset = 0

cap = None
net = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def offset_to_angle(offset, mean, direction, min_angle, max_angle):
    raw_angle = mean + direction * offset
    return clamp(raw_angle, min_angle, max_angle)


def apply_t1_safety(shoulder_offset, elbow_offset):
    if shoulder_offset <= T1_SHOULDER_LIMIT + T1_SHOULDER_BUFFER:
        elbow_offset = clamp(
            elbow_offset,
            T1_ELBOW_MIN_ALLOWED,
            ELBOW_OFFSET_MAX
        )
    else:
        elbow_offset = clamp(
            elbow_offset,
            ELBOW_OFFSET_MIN,
            ELBOW_OFFSET_MAX
        )

    return shoulder_offset, elbow_offset


def step_toward(current, target, max_step):
    diff = target - current

    if abs(diff) <= max_step:
        return target

    if diff > 0:
        return current + max_step

    return current - max_step


def write_all_motors(base_offset, shoulder_offset, elbow_offset):
    base_angle = offset_to_angle(
        base_offset,
        BASE_MEAN,
        BASE_DIRECTION,
        BASE_MIN,
        BASE_MAX
    )

    shoulder_angle = offset_to_angle(
        shoulder_offset,
        SHOULDER_MEAN,
        SHOULDER_DIRECTION,
        SHOULDER_MIN,
        SHOULDER_MAX
    )

    elbow_angle = offset_to_angle(
        elbow_offset,
        ELBOW_MEAN,
        ELBOW_DIRECTION,
        ELBOW_MIN,
        ELBOW_MAX
    )

    with motor_lock:
        kit.servo[BASE_CHANNEL].angle = base_angle
        kit.servo[SHOULDER_CHANNEL].angle = shoulder_angle
        kit.servo[ELBOW_CHANNEL].angle = elbow_angle


def disable_all_motors():
    with motor_lock:
        kit.servo[BASE_CHANNEL].angle = None
        kit.servo[SHOULDER_CHANNEL].angle = None
        kit.servo[ELBOW_CHANNEL].angle = None


def go_to_mean_position():
    global current_base_offset
    global current_shoulder_offset
    global current_elbow_offset

    print("Returning all motors to mean position...")

    while True:
        current_base_offset = step_toward(
            current_base_offset,
            0,
            BASE_MAX_STEP
        )

        current_shoulder_offset = step_toward(
            current_shoulder_offset,
            0,
            SHOULDER_MAX_STEP
        )

        current_elbow_offset = step_toward(
            current_elbow_offset,
            0,
            ELBOW_MAX_STEP
        )

        current_shoulder_offset, current_elbow_offset = apply_t1_safety(
            current_shoulder_offset,
            current_elbow_offset
        )

        write_all_motors(
            current_base_offset,
            current_shoulder_offset,
            current_elbow_offset
        )

        if (
            current_base_offset == 0
            and current_shoulder_offset == 0
            and current_elbow_offset == 0
        ):
            break

        time.sleep(0.02)

    print("All motors are at mean position.")


def find_largest_person(detections, frame_width, frame_height):
    best_box = None
    best_area = 0
    best_confidence = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        if class_id != PERSON_CLASS_ID:
            continue

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        box = detections[0, 0, i, 3:7] * [
            frame_width,
            frame_height,
            frame_width,
            frame_height
        ]

        x1, y1, x2, y2 = box.astype("int")

        x1 = int(clamp(x1, 0, frame_width - 1))
        y1 = int(clamp(y1, 0, frame_height - 1))
        x2 = int(clamp(x2, 0, frame_width - 1))
        y2 = int(clamp(y2, 0, frame_height - 1))

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width <= 0 or box_height <= 0:
            continue

        area = box_width * box_height

        if area > best_area:
            best_area = area
            best_box = (x1, y1, x2, y2)
            best_confidence = confidence

    return best_box, best_confidence


def apply_camera_orientation(frame):
    if CAMERA_ROTATION is not None:
        frame = cv2.rotate(frame, CAMERA_ROTATION)

    return frame


# ============================================================
# CAMERA + AI + MOTOR LOOP
# The shoulder/elbow tracking equations below are preserved from
# the uploaded working file.
# ============================================================

def tracking_loop():
    global latest_jpeg
    global camera_running
    global person_detected
    global current_confidence
    global fps

    global current_base_offset
    global current_shoulder_offset
    global current_elbow_offset

    global target_base_offset
    global target_shoulder_offset
    global target_elbow_offset

    frame_counter = 0
    fps_start_time = time.time()

    while not shutdown_event.is_set():
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame.")
            camera_running = False
            time.sleep(0.1)
            continue

        camera_running = True

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = apply_camera_orientation(frame)

        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=0.007843,
            size=(300, 300),
            mean=127.5
        )

        net.setInput(blob)
        detections = net.forward()

        person_box, confidence = find_largest_person(
            detections,
            w,
            h
        )

        with state_lock:
            person_detected = person_box is not None
            current_confidence = float(confidence)

        if person_box is not None:
            x1, y1, x2, y2 = person_box

            person_center_x = (x1 + x2) // 2
            person_center_y = (y1 + y2) // 2
            person_height = y2 - y1

            frame_center_x = w // 2
            frame_center_y = h // 2

            x_error = (person_center_x - frame_center_x) / (w / 2)
            y_error = (person_center_y - frame_center_y) / (h / 2)

            person_height_ratio = person_height / h
            size_error = TARGET_PERSON_HEIGHT_RATIO - person_height_ratio

            # BASE TRACKING - unchanged
            if abs(x_error) > X_DEADZONE:
                target_base_offset += (
                    BASE_TRACK_SIGN
                    * x_error
                    * BASE_GAIN
                )

            # SHOULDER TRACKING - exact original logic
            shoulder_error = y_error + (0.4 * size_error)

            if abs(shoulder_error) > Y_DEADZONE:
                target_shoulder_offset += (
                    SHOULDER_TRACK_SIGN
                    * shoulder_error
                    * SHOULDER_GAIN
                )

            # ELBOW TRACKING - exact original logic
            elbow_error = size_error + (0.5 * y_error)

            if abs(elbow_error) > SIZE_DEADZONE:
                target_elbow_offset += (
                    ELBOW_TRACK_SIGN
                    * elbow_error
                    * ELBOW_GAIN
                )

            target_base_offset = clamp(
                target_base_offset,
                BASE_OFFSET_MIN,
                BASE_OFFSET_MAX
            )

            target_shoulder_offset = clamp(
                target_shoulder_offset,
                SHOULDER_OFFSET_MIN,
                SHOULDER_OFFSET_MAX
            )

            target_elbow_offset = clamp(
                target_elbow_offset,
                ELBOW_OFFSET_MIN,
                ELBOW_OFFSET_MAX
            )

            target_shoulder_offset, target_elbow_offset = apply_t1_safety(
                target_shoulder_offset,
                target_elbow_offset
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Person: {confidence * 100:.1f}%",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (person_center_x, person_center_y),
                5,
                (0, 0, 255),
                -1
            )

            cv2.circle(
                frame,
                (frame_center_x, frame_center_y),
                5,
                (255, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                f"x:{x_error:.2f} y:{y_error:.2f} size:{size_error:.2f}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                f"shoulder_err:{shoulder_error:.2f} elbow_err:{elbow_error:.2f}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

        else:
            cv2.putText(
                frame,
                "No person detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # FAST COORDINATED MOTOR MOVEMENT - unchanged
        current_base_offset = step_toward(
            current_base_offset,
            target_base_offset,
            BASE_MAX_STEP
        )

        current_shoulder_offset = step_toward(
            current_shoulder_offset,
            target_shoulder_offset,
            SHOULDER_MAX_STEP
        )

        current_elbow_offset = step_toward(
            current_elbow_offset,
            target_elbow_offset,
            ELBOW_MAX_STEP
        )

        current_shoulder_offset, current_elbow_offset = apply_t1_safety(
            current_shoulder_offset,
            current_elbow_offset
        )

        write_all_motors(
            current_base_offset,
            current_shoulder_offset,
            current_elbow_offset
        )

        frame_counter += 1
        elapsed_time = time.time() - fps_start_time

        if elapsed_time >= 1.0:
            fps = frame_counter / elapsed_time
            frame_counter = 0
            fps_start_time = time.time()

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, h - 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Base offset: {current_base_offset:.1f}",
            (20, h - 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Shoulder offset: {current_shoulder_offset:.1f}",
            (20, h - 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Elbow offset: {current_elbow_offset:.1f}",
            (20, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            (
                f"Targets B/S/E: {target_base_offset:.1f}, "
                f"{target_shoulder_offset:.1f}, "
                f"{target_elbow_offset:.1f}"
            ),
            (20, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        if current_shoulder_offset <= T1_SHOULDER_LIMIT + T1_SHOULDER_BUFFER:
            cv2.putText(
                frame,
                "T1 SAFETY ACTIVE",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        encode_ok, jpeg_buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )

        if encode_ok:
            with frame_lock:
                latest_jpeg = jpeg_buffer.tobytes()


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


def generate_video_stream():
    while not shutdown_event.is_set():
        with frame_lock:
            frame = latest_jpeg

        if frame is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )

        time.sleep(0.01)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    t1_active = (
        current_shoulder_offset
        <= T1_SHOULDER_LIMIT + T1_SHOULDER_BUFFER
    )

    with state_lock:
        return jsonify({
            "hostname": socket.gethostname(),
            "camera_running": camera_running,
            "person_detected": person_detected,
            "confidence": current_confidence * 100.0,
            "fps": fps,
            "base_offset": current_base_offset,
            "shoulder_offset": current_shoulder_offset,
            "elbow_offset": current_elbow_offset,
            "t1_active": t1_active
        })


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "camera_running": camera_running
    })


# ============================================================
# STARTUP / SHUTDOWN
# ============================================================

def handle_shutdown(signum, frame):
    print(f"Shutdown signal received: {signum}")
    shutdown_event.set()


def initialize_system():
    global cap
    global net
    global camera_running

    print("Moving all motors to mean position...")
    write_all_motors(0, 0, 0)
    time.sleep(1)

    print("Loading MobileNet SSD model...")

    if not os.path.exists(PROTOTXT_PATH):
        raise FileNotFoundError(
            f"Could not find prototxt file: {PROTOTXT_PATH}"
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find model file: {MODEL_PATH}"
        )

    net = cv2.dnn.readNetFromCaffe(
        PROTOTXT_PATH,
        MODEL_PATH
    )

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_V4L2
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        CAMERA_FPS
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    camera_running = True
    print("Camera opened.")


def cleanup_system():
    print("Stopping robot arm system...")

    if cap is not None:
        cap.release()

    try:
        go_to_mean_position()
        time.sleep(0.5)
        disable_all_motors()
        print("Motors returned to mean position and cut off.")
    except Exception as error:
        print(f"Motor cleanup error: {error}")


def main():
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    tracking_thread = None

    try:
        initialize_system()

        tracking_thread = threading.Thread(
            target=tracking_loop,
            name="TrackingThread",
            daemon=True
        )

        tracking_thread.start()

        hostname = socket.gethostname()

        print()
        print("Robot arm web server is running.")
        print(f"Open: http://{hostname}.local:{WEB_PORT}")
        print()

        server = make_server(
            WEB_HOST,
            WEB_PORT,
            app,
            threaded=True
        )

        server.timeout = 1.0

        while not shutdown_event.is_set():
            server.handle_request()

    except KeyboardInterrupt:
        shutdown_event.set()

    finally:
        shutdown_event.set()

        if tracking_thread is not None:
            tracking_thread.join(timeout=3)

        cleanup_system()


if __name__ == "__main__":
    main()
