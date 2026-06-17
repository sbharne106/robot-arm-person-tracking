#!/usr/bin/env python3

import os
import cv2
import time
import signal
import socket
import threading
import traceback

from flask import Flask, Response, jsonify, render_template_string
from werkzeug.serving import make_server
from adafruit_servokit import ServoKit


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIRECTORY = os.path.join(
    PROJECT_DIR,
    "mobilenet_ssd"
)

PROTOTXT_PATH = os.path.join(
    MODEL_DIRECTORY,
    "MobileNetSSD_deploy.prototxt"
)

MODEL_PATH = os.path.join(
    MODEL_DIRECTORY,
    "MobileNetSSD_deploy.caffemodel"
)


# ============================================================
# CAMERA SETTINGS
# ============================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30

JPEG_QUALITY = 80

USE_MJPG = True

FLIP_HORIZONTAL = False
FLIP_VERTICAL = False

# Rotation choices:
#
# None
# cv2.ROTATE_90_CLOCKWISE
# cv2.ROTATE_90_COUNTERCLOCKWISE
# cv2.ROTATE_180
CAMERA_ROTATION = cv2.ROTATE_90_CLOCKWISE


# ============================================================
# WEB SERVER SETTINGS
# ============================================================

WEB_HOST = "0.0.0.0"
WEB_PORT = 5000


# ============================================================
# AI MODEL SETTINGS
# ============================================================

PERSON_CLASS_ID = 15
CONFIDENCE_THRESHOLD = 0.45

DNN_INPUT_WIDTH = 300
DNN_INPUT_HEIGHT = 300


# ============================================================
# PCA9685 SETTINGS
# ============================================================

PCA9685_ADDRESS = 0x40
PCA9685_CHANNELS = 16

BASE_CHANNEL = 3
ELBOW_CHANNEL = 2
SHOULDER_CHANNEL = 0

SERVO_ACTUATION_RANGE = 270

SERVO_MIN_PULSE = 500
SERVO_MAX_PULSE = 2500


# ============================================================
# MOTOR CALIBRATION
# Physical ServoKit angles
# ============================================================

# Base
BASE_MIN = 0.0
BASE_MEAN = 130.0
BASE_MAX = 270.0

# Shoulder
SHOULDER_MIN = 30.0
SHOULDER_MEAN = 190.0
SHOULDER_MAX = 270.0

# Elbow
ELBOW_MIN = 100.0
ELBOW_MEAN = 240.0
ELBOW_MAX = 260.0

# Forward/back coordinated movement
DISTANCE_SHOULDER_GAIN = 7.0
DISTANCE_ELBOW_GAIN = 8.0

DISTANCE_SHOULDER_MAX_STEP = 4.0
DISTANCE_ELBOW_MAX_STEP = 5.0

# Reverse either value if that motor moves the wrong way.
SHOULDER_DISTANCE_SIGN = -1.0
ELBOW_DISTANCE_SIGN = 1.0
# ============================================================
# TRACKING DIRECTION
#
# Change a sign from 1 to -1 if that motor moves in the wrong
# direction.
# ============================================================

BASE_TRACK_SIGN = -1.0
SHOULDER_TRACK_SIGN = -1.0
ELBOW_TRACK_SIGN = 1.0


# ============================================================
# TRACKING SETTINGS
# ============================================================

# Horizontal person-position tracking
HORIZONTAL_DEAD_ZONE_RATIO = 0.08

# Vertical person-position tracking
VERTICAL_DEAD_ZONE_RATIO = 0.08

# Desired person-box height relative to the image height
TARGET_PERSON_HEIGHT_RATIO = 0.55

# Distance dead zone
HEIGHT_DEAD_ZONE_RATIO = 0.03


# ============================================================
# MOTOR SPEED SETTINGS
#
# Higher gain produces a stronger response.
# Higher max step allows faster motion per frame.
# ============================================================

BASE_GAIN = 5.0
SHOULDER_GAIN = 6.0
ELBOW_GAIN = 8.0

BASE_MAX_STEP = 10.0
SHOULDER_MAX_STEP = 8.0
ELBOW_MAX_STEP = 8.0

MOTOR_UPDATE_INTERVAL = 0.02


# ============================================================
# SMOOTH RETURN-TO-MEAN SETTINGS
# ============================================================

RETURN_MAX_STEP = 2.0
RETURN_STEP_DELAY = 0.025


# ============================================================
# SAFETY CONDITION T1
#
# Your offset calibration:
#
# Shoulder mean = 190
# Shoulder offset -160 = physical angle 30
#
# Elbow mean = 240
# Elbow is reversed
# Elbow offset 50 = physical angle 190
#
# Therefore, when the shoulder reaches approximately 30 degrees,
# the elbow is prevented from moving above physical angle 190.
# ============================================================

T1_SHOULDER_TRIGGER_ANGLE = 35.0
T1_ELBOW_MAX_ANGLE = 190.0


# ============================================================
# GLOBAL OBJECTS AND STATE
# ============================================================

app = Flask(__name__)

shutdown_event = threading.Event()

frame_lock = threading.Lock()
state_lock = threading.Lock()
motor_lock = threading.Lock()

camera = None
net = None
kit = None
web_server = None

latest_jpeg = None

camera_running = False
model_running = False
tracking_enabled = True
person_detected = False

current_fps = 0.0
current_confidence = 0.0
current_command = "STARTING"

base_angle = BASE_MEAN
shoulder_angle = SHOULDER_MEAN
elbow_angle = ELBOW_MEAN

last_motor_update_time = 0.0


# ============================================================
# WEB PAGE
# ============================================================

HTML_PAGE = """
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

        html,
        body {
            margin: 0;
            padding: 0;

            width: 100%;
            min-height: 100%;

            background-color: #111827;
            color: white;

            font-family:
                Arial,
                Helvetica,
                sans-serif;
        }

        body {
            min-height: 100vh;
        }

        .container {
            width: 95%;
            max-width: 1050px;

            margin: 0 auto;
            padding: 22px 0 30px 0;

            text-align: center;
        }

        h1 {
            margin: 0 0 6px 0;

            font-size: clamp(
                25px,
                5vw,
                40px
            );
        }

        .subtitle {
            margin-bottom: 18px;
            color: #9ca3af;
        }

        .video-wrapper {
            width: 100%;
            margin: 0;
            padding: 0;

            overflow: hidden;

            border: 2px solid #374151;
            border-radius: 12px;

            background-color: black;
            line-height: 0;
        }

        #camera-feed {
            display: block;

            width: 100%;
            height: auto;

            margin: 0;
            padding: 0;

            background-color: black;
        }

        .status-grid {
            display: grid;

            grid-template-columns:
                repeat(
                    auto-fit,
                    minmax(140px, 1fr)
                );

            gap: 10px;
            margin-top: 16px;
        }

        .status-card {
            padding: 13px 10px;

            border: 1px solid #374151;
            border-radius: 10px;

            background-color: #1f2937;
        }

        .label {
            margin-bottom: 5px;

            color: #9ca3af;
            font-size: 13px;
        }

        .value {
            font-size: 17px;
            font-weight: bold;
        }

        .controls {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;

            gap: 10px;
            margin-top: 18px;
        }

        button {
            min-width: 150px;

            padding: 12px 18px;

            border: none;
            border-radius: 8px;

            font-size: 15px;
            font-weight: bold;

            cursor: pointer;
        }

        .start-button {
            background-color: #22c55e;
            color: #052e16;
        }

        .pause-button {
            background-color: #f59e0b;
            color: #451a03;
        }

        .mean-button {
            background-color: #60a5fa;
            color: #172554;
        }

        button:active {
            transform: scale(0.98);
        }

        .connection {
            margin-top: 14px;

            color: #9ca3af;
            font-size: 13px;
        }

        .online {
            color: #4ade80;
        }

        .offline {
            color: #f87171;
        }

    </style>

</head>

<body>

    <div class="container">

        <h1>Robot Arm Person Tracking</h1>

        <div class="subtitle">
            Live camera, human detection and motor tracking
        </div>

        <div class="video-wrapper">

            <img
                id="camera-feed"
                src="/video_feed"
                alt="Robot arm live camera feed"
            >

        </div>

        <div class="status-grid">

            <div class="status-card">
                <div class="label">System</div>
                <div class="value" id="system-status">
                    Connecting
                </div>
            </div>

            <div class="status-card">
                <div class="label">Tracking</div>
                <div class="value" id="tracking-status">
                    --
                </div>
            </div>

            <div class="status-card">
                <div class="label">Person</div>
                <div class="value" id="person-status">
                    --
                </div>
            </div>

            <div class="status-card">
                <div class="label">Command</div>
                <div class="value" id="command">
                    --
                </div>
            </div>

            <div class="status-card">
                <div class="label">Confidence</div>
                <div class="value" id="confidence">
                    --
                </div>
            </div>

            <div class="status-card">
                <div class="label">FPS</div>
                <div class="value" id="fps">
                    --
                </div>
            </div>

            <div class="status-card">
                <div class="label">Base</div>
                <div class="value" id="base-angle">
                    --
                </div>
            </div>

            <div class="status-card">
                <div class="label">Shoulder</div>
                <div class="value" id="shoulder-angle">
                    --
                </div>
            </div>

            <div class="status-card">
                <div class="label">Elbow</div>
                <div class="value" id="elbow-angle">
                    --
                </div>
            </div>

        </div>

        <div class="controls">

            <button
                class="start-button"
                onclick="sendCommand('/tracking/start')"
            >
                Start Tracking
            </button>

            <button
                class="pause-button"
                onclick="sendCommand('/tracking/pause')"
            >
                Pause Tracking
            </button>

            <button
                class="mean-button"
                onclick="sendCommand('/motors/mean')"
            >
                Return to Mean
            </button>

        </div>

        <div
            class="connection"
            id="connection-message"
        >
            Connecting to Raspberry Pi...
        </div>

    </div>

    <script>

        async function sendCommand(url) {

            try {

                const response = await fetch(
                    url,
                    {
                        method: "POST"
                    }
                );

                if (!response.ok) {
                    throw new Error("Command failed");
                }

                await updateStatus();

            } catch (error) {

                console.error(error);

                const message =
                    document.getElementById(
                        "connection-message"
                    );

                message.textContent =
                    "Command could not be sent";

                message.className =
                    "connection offline";
            }
        }


        async function updateStatus() {

            const connectionMessage =
                document.getElementById(
                    "connection-message"
                );

            try {

                const response = await fetch(
                    "/status",
                    {
                        cache: "no-store"
                    }
                );

                if (!response.ok) {
                    throw new Error("Status request failed");
                }

                const data = await response.json();

                document.getElementById(
                    "system-status"
                ).textContent =
                    data.camera_running
                        && data.model_running
                        ? "ONLINE"
                        : "STARTING";

                document.getElementById(
                    "tracking-status"
                ).textContent =
                    data.tracking_enabled
                        ? "ENABLED"
                        : "PAUSED";

                document.getElementById(
                    "person-status"
                ).textContent =
                    data.person_detected
                        ? "DETECTED"
                        : "NOT DETECTED";

                document.getElementById(
                    "command"
                ).textContent =
                    data.command;

                document.getElementById(
                    "confidence"
                ).textContent =
                    data.confidence.toFixed(1)
                    + "%";

                document.getElementById(
                    "fps"
                ).textContent =
                    data.fps.toFixed(1);

                document.getElementById(
                    "base-angle"
                ).textContent =
                    data.base_angle.toFixed(1)
                    + "°";

                document.getElementById(
                    "shoulder-angle"
                ).textContent =
                    data.shoulder_angle.toFixed(1)
                    + "°";

                document.getElementById(
                    "elbow-angle"
                ).textContent =
                    data.elbow_angle.toFixed(1)
                    + "°";

                connectionMessage.textContent =
                    "Connected to "
                    + data.hostname;

                connectionMessage.className =
                    "connection online";

            } catch (error) {

                document.getElementById(
                    "system-status"
                ).textContent =
                    "OFFLINE";

                connectionMessage.textContent =
                    "Unable to reach Raspberry Pi";

                connectionMessage.className =
                    "connection offline";
            }
        }


        updateStatus();

        setInterval(
            updateStatus,
            1000
        );

    </script>

</body>

</html>
"""


# ============================================================
# GENERAL HELPERS
# ============================================================

def clamp(value, minimum, maximum):
    return max(
        minimum,
        min(maximum, value)
    )


def move_toward(
    current_value,
    target_value,
    maximum_step
):
    difference = target_value - current_value

    if abs(difference) <= maximum_step:
        return target_value

    if difference > 0:
        return current_value + maximum_step

    return current_value - maximum_step


def calculate_tracking_step(
    normalized_error,
    gain,
    maximum_step
):
    movement = normalized_error * gain

    return clamp(
        movement,
        -maximum_step,
        maximum_step
    )


# ============================================================
# CAMERA ORIENTATION
# ============================================================

def apply_camera_orientation(frame):

    if FLIP_HORIZONTAL:
        frame = cv2.flip(
            frame,
            1
        )

    if FLIP_VERTICAL:
        frame = cv2.flip(
            frame,
            0
        )

    if CAMERA_ROTATION is not None:
        frame = cv2.rotate(
            frame,
            CAMERA_ROTATION
        )

    return frame


# ============================================================
# MOTOR SETUP
# ============================================================

def initialize_motors():

    global kit
    global base_angle
    global shoulder_angle
    global elbow_angle

    print("Initializing PCA9685 and servos...")

    kit = ServoKit(
        channels=PCA9685_CHANNELS,
        address=PCA9685_ADDRESS
    )

    servo_channels = [
        BASE_CHANNEL,
        SHOULDER_CHANNEL,
        ELBOW_CHANNEL
    ]

    for channel in servo_channels:

        kit.servo[
            channel
        ].actuation_range = (
            SERVO_ACTUATION_RANGE
        )

        kit.servo[
            channel
        ].set_pulse_width_range(
            SERVO_MIN_PULSE,
            SERVO_MAX_PULSE
        )

    base_angle = BASE_MEAN
    shoulder_angle = SHOULDER_MEAN
    elbow_angle = ELBOW_MEAN

    set_motor_angles(
        BASE_MEAN,
        SHOULDER_MEAN,
        ELBOW_MEAN
    )

    print("Motors moved to mean positions.")


# ============================================================
# MOTOR SAFETY
# ============================================================

def apply_motor_safety(
    requested_base,
    requested_shoulder,
    requested_elbow
):

    safe_base = clamp(
        requested_base,
        BASE_MIN,
        BASE_MAX
    )

    safe_shoulder = clamp(
        requested_shoulder,
        SHOULDER_MIN,
        SHOULDER_MAX
    )

    safe_elbow = clamp(
        requested_elbow,
        ELBOW_MIN,
        ELBOW_MAX
    )

    # T1 safety:
    # When shoulder reaches its lowest region, prevent the elbow
    # from moving farther into the unsafe area.
    if (
        safe_shoulder
        <= T1_SHOULDER_TRIGGER_ANGLE
    ):
        safe_elbow = min(
            safe_elbow,
            T1_ELBOW_MAX_ANGLE
        )

    return (
        safe_base,
        safe_shoulder,
        safe_elbow
    )


# ============================================================
# WRITE MOTOR ANGLES
# ============================================================

def set_motor_angles(
    requested_base,
    requested_shoulder,
    requested_elbow
):

    global base_angle
    global shoulder_angle
    global elbow_angle

    if kit is None:
        return

    (
        safe_base,
        safe_shoulder,
        safe_elbow
    ) = apply_motor_safety(
        requested_base,
        requested_shoulder,
        requested_elbow
    )

    with motor_lock:

        kit.servo[
            BASE_CHANNEL
        ].angle = safe_base

        kit.servo[
            SHOULDER_CHANNEL
        ].angle = safe_shoulder

        kit.servo[
            ELBOW_CHANNEL
        ].angle = safe_elbow

        base_angle = safe_base
        shoulder_angle = safe_shoulder
        elbow_angle = safe_elbow


# ============================================================
# RETURN MOTORS TO MEAN
# ============================================================

def return_motors_to_mean():

    global current_command

    if kit is None:
        return

    with state_lock:
        current_command = "RETURNING TO MEAN"

    print("Returning motors to mean positions...")

    while not shutdown_event.is_set():

        base_finished = (
            abs(
                base_angle - BASE_MEAN
            ) < 0.5
        )

        shoulder_finished = (
            abs(
                shoulder_angle - SHOULDER_MEAN
            ) < 0.5
        )

        elbow_finished = (
            abs(
                elbow_angle - ELBOW_MEAN
            ) < 0.5
        )

        if (
            base_finished
            and shoulder_finished
            and elbow_finished
        ):
            break

        next_base = move_toward(
            base_angle,
            BASE_MEAN,
            RETURN_MAX_STEP
        )

        next_shoulder = move_toward(
            shoulder_angle,
            SHOULDER_MEAN,
            RETURN_MAX_STEP
        )

        next_elbow = move_toward(
            elbow_angle,
            ELBOW_MEAN,
            RETURN_MAX_STEP
        )

        set_motor_angles(
            next_base,
            next_shoulder,
            next_elbow
        )

        time.sleep(
            RETURN_STEP_DELAY
        )

    set_motor_angles(
        BASE_MEAN,
        SHOULDER_MEAN,
        ELBOW_MEAN
    )

    with state_lock:
        current_command = "AT MEAN"

    print("Motors are at mean positions.")


def return_motors_to_mean_during_shutdown():
    """
    The normal function exits when shutdown_event is set, so this
    separate function is used during final cleanup.
    """

    global current_command

    if kit is None:
        return

    with state_lock:
        current_command = "SHUTDOWN: RETURNING TO MEAN"

    print("Shutdown: returning motors to mean...")

    while True:

        base_finished = (
            abs(
                base_angle - BASE_MEAN
            ) < 0.5
        )

        shoulder_finished = (
            abs(
                shoulder_angle - SHOULDER_MEAN
            ) < 0.5
        )

        elbow_finished = (
            abs(
                elbow_angle - ELBOW_MEAN
            ) < 0.5
        )

        if (
            base_finished
            and shoulder_finished
            and elbow_finished
        ):
            break

        next_base = move_toward(
            base_angle,
            BASE_MEAN,
            RETURN_MAX_STEP
        )

        next_shoulder = move_toward(
            shoulder_angle,
            SHOULDER_MEAN,
            RETURN_MAX_STEP
        )

        next_elbow = move_toward(
            elbow_angle,
            ELBOW_MEAN,
            RETURN_MAX_STEP
        )

        set_motor_angles(
            next_base,
            next_shoulder,
            next_elbow
        )

        time.sleep(
            RETURN_STEP_DELAY
        )

    set_motor_angles(
        BASE_MEAN,
        SHOULDER_MEAN,
        ELBOW_MEAN
    )

    print("Shutdown: motors returned to mean.")


# ============================================================
# AI MODEL SETUP
# ============================================================

def initialize_model():

    global net
    global model_running

    if not os.path.isfile(
        PROTOTXT_PATH
    ):
        raise FileNotFoundError(
            "MobileNet SSD prototxt not found: "
            + PROTOTXT_PATH
        )

    if not os.path.isfile(
        MODEL_PATH
    ):
        raise FileNotFoundError(
            "MobileNet SSD model not found: "
            + MODEL_PATH
        )

    print("Loading MobileNet SSD model...")

    net = cv2.dnn.readNetFromCaffe(
        PROTOTXT_PATH,
        MODEL_PATH
    )

    model_running = True

    print("MobileNet SSD model loaded.")


# ============================================================
# CAMERA SETUP
# ============================================================

def initialize_camera():

    global camera
    global camera_running

    print(
        f"Opening camera index {CAMERA_INDEX}..."
    )

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_V4L2
    )

    if USE_MJPG:

        camera.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(
                *"MJPG"
            )
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
        CAMERA_FPS
    )

    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open camera index "
            + str(CAMERA_INDEX)
        )

    # Allow exposure and white balance to settle.
    successful_frames = 0

    for _ in range(20):

        success, frame = camera.read()

        if success and frame is not None:
            successful_frames += 1

        time.sleep(0.03)

    if successful_frames == 0:
        raise RuntimeError(
            "Camera opened but no frames were received."
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

    camera_running = True

    print(
        "Camera ready: "
        f"{actual_width}x{actual_height}, "
        f"reported FPS {reported_fps:.1f}"
    )


# ============================================================
# PERSON DETECTION
# ============================================================

def detect_largest_person(frame):

    frame_height, frame_width = (
        frame.shape[:2]
    )

    resized_frame = cv2.resize(
        frame,
        (
            DNN_INPUT_WIDTH,
            DNN_INPUT_HEIGHT
        )
    )

    blob = cv2.dnn.blobFromImage(
        resized_frame,
        scalefactor=0.007843,
        size=(
            DNN_INPUT_WIDTH,
            DNN_INPUT_HEIGHT
        ),
        mean=127.5
    )

    net.setInput(blob)

    detections = net.forward()

    best_box = None
    best_confidence = 0.0
    best_area = 0

    number_of_detections = (
        detections.shape[2]
    )

    for detection_index in range(
        number_of_detections
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

        if class_id != PERSON_CLASS_ID:
            continue

        if (
            confidence
            < CONFIDENCE_THRESHOLD
        ):
            continue

        box = detections[
            0,
            0,
            detection_index,
            3:7
        ] * [
            frame_width,
            frame_height,
            frame_width,
            frame_height
        ]

        x1, y1, x2, y2 = (
            box.astype(int)
        )

        x1 = int(
            clamp(
                x1,
                0,
                frame_width - 1
            )
        )

        y1 = int(
            clamp(
                y1,
                0,
                frame_height - 1
            )
        )

        x2 = int(
            clamp(
                x2,
                0,
                frame_width - 1
            )
        )

        y2 = int(
            clamp(
                y2,
                0,
                frame_height - 1
            )
        )

        box_width = max(
            0,
            x2 - x1
        )

        box_height = max(
            0,
            y2 - y1
        )

        box_area = (
            box_width
            * box_height
        )

        if box_area > best_area:

            best_area = box_area
            best_confidence = confidence

            best_box = (
                x1,
                y1,
                x2,
                y2
            )

    return (
        best_box,
        best_confidence
    )


# ============================================================
# PERSON TRACKING MOTOR LOGIC
# ============================================================

def update_motor_tracking(
    person_box,
    frame_width,
    frame_height
):

    global current_command
    global last_motor_update_time

    if person_box is None:

        with state_lock:
            current_command = "NO PERSON"

        return

    current_time = time.monotonic()

    if (
        current_time
        - last_motor_update_time
        < MOTOR_UPDATE_INTERVAL
    ):
        return

    last_motor_update_time = current_time

    x1, y1, x2, y2 = person_box

    person_center_x = (
        x1 + x2
    ) / 2.0

    person_center_y = (
        y1 + y2
    ) / 2.0

    person_height = (
        y2 - y1
    )

    frame_center_x = (
        frame_width / 2.0
    )

    frame_center_y = (
        frame_height / 2.0
    )

    horizontal_error = (
        person_center_x
        - frame_center_x
    ) / frame_center_x

    vertical_error = (
        person_center_y
        - frame_center_y
    ) / frame_center_y

    person_height_ratio = (
        person_height
        / float(frame_height)
    )

    distance_error = (
        TARGET_PERSON_HEIGHT_RATIO
        - person_height_ratio
    )

    requested_base = base_angle
    requested_shoulder = shoulder_angle
    requested_elbow = elbow_angle

    command_parts = []

    # --------------------------------------------------------
    # BASE: left and right
    # --------------------------------------------------------

    if (
        abs(horizontal_error)
        > HORIZONTAL_DEAD_ZONE_RATIO
    ):

        base_step = calculate_tracking_step(
            horizontal_error,
            BASE_GAIN,
            BASE_MAX_STEP
        )

        requested_base += (
            BASE_TRACK_SIGN
            * base_step
        )

        if horizontal_error < 0:
            command_parts.append(
                "PERSON LEFT"
            )
        else:
            command_parts.append(
                "PERSON RIGHT"
            )

    else:
        command_parts.append(
            "HORIZONTAL CENTER"
        )

    # --------------------------------------------------------
    # SHOULDER: person above or below frame center
    # --------------------------------------------------------

    if (
        abs(vertical_error)
        > VERTICAL_DEAD_ZONE_RATIO
    ):

        shoulder_step = (
            calculate_tracking_step(
                vertical_error,
                SHOULDER_GAIN,
                SHOULDER_MAX_STEP
            )
        )

        requested_shoulder += (
            SHOULDER_TRACK_SIGN
            * shoulder_step
        )

        if vertical_error < 0:
            command_parts.append(
                "PERSON HIGH"
            )
        else:
            command_parts.append(
                "PERSON LOW"
            )

    else:
        command_parts.append(
            "VERTICAL CENTER"
        )

    # --------------------------------------------------------
    # ELBOW: person distance based on bounding-box height
    # --------------------------------------------------------

    if (
        abs(distance_error)
        > HEIGHT_DEAD_ZONE_RATIO
    ):

        elbow_step = (
            calculate_tracking_step(
                distance_error,
                ELBOW_GAIN,
                ELBOW_MAX_STEP
            )
        )

        requested_elbow += (
            ELBOW_TRACK_SIGN
            * elbow_step
        )

        if distance_error > 0:
            command_parts.append(
                "PERSON FAR"
            )
        else:
            command_parts.append(
                "PERSON CLOSE"
            )

    else:
        command_parts.append(
            "DISTANCE OK"
        )

    set_motor_angles(
        requested_base,
        requested_shoulder,
        requested_elbow
    )

    with state_lock:
        current_command = (
            " | ".join(
                command_parts
            )
        )


# ============================================================
# DRAW CAMERA OVERLAY
# ============================================================

def draw_tracking_overlay(
    frame,
    person_box,
    confidence,
    fps
):

    frame_height, frame_width = (
        frame.shape[:2]
    )

    frame_center_x = (
        frame_width // 2
    )

    frame_center_y = (
        frame_height // 2
    )

    horizontal_dead_zone = int(
        frame_width
        * HORIZONTAL_DEAD_ZONE_RATIO
    )

    vertical_dead_zone = int(
        frame_height
        * VERTICAL_DEAD_ZONE_RATIO
    )

    # Center tracking box
    cv2.rectangle(
        frame,
        (
            frame_center_x
            - horizontal_dead_zone,
            frame_center_y
            - vertical_dead_zone
        ),
        (
            frame_center_x
            + horizontal_dead_zone,
            frame_center_y
            + vertical_dead_zone
        ),
        (255, 255, 0),
        1
    )

    if person_box is not None:

        x1, y1, x2, y2 = person_box

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        person_center_x = int(
            (x1 + x2) / 2
        )

        person_center_y = int(
            (y1 + y2) / 2
        )

        cv2.circle(
            frame,
            (
                person_center_x,
                person_center_y
            ),
            5,
            (0, 0, 255),
            -1
        )

        person_label = (
            "Person: "
            f"{confidence * 100:.1f}%"
        )

        cv2.putText(
            frame,
            person_label,
            (
                x1,
                max(
                    25,
                    y1 - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    with state_lock:

        command_text = (
            current_command
        )

        tracking_text = (
            "ENABLED"
            if tracking_enabled
            else "PAUSED"
        )

    # Top information area
    cv2.rectangle(
        frame,
        (0, 0),
        (
            frame_width,
            88
        ),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Tracking: "
        + tracking_text,
        (12, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        command_text,
        (12, 77),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1
    )

    # Bottom motor information area
    cv2.rectangle(
        frame,
        (
            0,
            frame_height - 36
        ),
        (
            frame_width,
            frame_height
        ),
        (0, 0, 0),
        -1
    )

    motor_text = (
        f"Base: {base_angle:.0f}  "
        f"Shoulder: {shoulder_angle:.0f}  "
        f"Elbow: {elbow_angle:.0f}"
    )

    cv2.putText(
        frame,
        motor_text,
        (
            12,
            frame_height - 11
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    return frame


# ============================================================
# MAIN CAMERA, AI AND TRACKING LOOP
# ============================================================

def camera_tracking_loop():

    global latest_jpeg
    global camera_running
    global person_detected
    global current_confidence
    global current_fps
    global current_command

    previous_frame_time = (
        time.perf_counter()
    )

    smoothed_fps = 0.0
    fps_smoothing = 0.10

    consecutive_failures = 0

    print(
        "Camera and tracking thread started."
    )

    while not shutdown_event.is_set():

        success, frame = camera.read()

        if not success or frame is None:

            consecutive_failures += 1

            print(
                "Camera frame read failed: "
                + str(
                    consecutive_failures
                )
            )

            if consecutive_failures >= 20:
                camera_running = False

            time.sleep(0.1)
            continue

        consecutive_failures = 0
        camera_running = True

        frame = apply_camera_orientation(
            frame
        )

        frame_height, frame_width = (
            frame.shape[:2]
        )

        try:

            (
                person_box,
                confidence
            ) = detect_largest_person(
                frame
            )

        except Exception as error:

            print(
                "Person detection error: "
                + str(error)
            )

            person_box = None
            confidence = 0.0

        with state_lock:

            person_detected = (
                person_box is not None
            )

            current_confidence = (
                confidence
            )

        if tracking_enabled:

            update_motor_tracking(
                person_box,
                frame_width,
                frame_height
            )

        else:

            with state_lock:
                current_command = (
                    "TRACKING PAUSED"
                )

        current_frame_time = (
            time.perf_counter()
        )

        elapsed_time = (
            current_frame_time
            - previous_frame_time
        )

        previous_frame_time = (
            current_frame_time
        )

        if elapsed_time > 0:

            instantaneous_fps = (
                1.0 / elapsed_time
            )

            if smoothed_fps == 0.0:

                smoothed_fps = (
                    instantaneous_fps
                )

            else:

                smoothed_fps = (
                    (
                        1.0
                        - fps_smoothing
                    )
                    * smoothed_fps
                    + fps_smoothing
                    * instantaneous_fps
                )

        current_fps = smoothed_fps

        display_frame = (
            draw_tracking_overlay(
                frame.copy(),
                person_box,
                confidence,
                current_fps
            )
        )

        encode_success, jpeg_buffer = (
            cv2.imencode(
                ".jpg",
                display_frame,
                [
                    int(
                        cv2.IMWRITE_JPEG_QUALITY
                    ),
                    JPEG_QUALITY
                ]
            )
        )

        if not encode_success:
            continue

        with frame_lock:
            latest_jpeg = (
                jpeg_buffer.tobytes()
            )

    print(
        "Camera and tracking thread stopped."
    )


# ============================================================
# WEB ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template_string(
        HTML_PAGE
    )


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
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


@app.route("/status")
def status():

    with state_lock:

        status_data = {
            "hostname": socket.gethostname(),
            "camera_running": camera_running,
            "model_running": model_running,
            "tracking_enabled": tracking_enabled,
            "person_detected": person_detected,
            "confidence": (
                current_confidence
                * 100.0
            ),
            "fps": current_fps,
            "command": current_command,
            "base_angle": base_angle,
            "shoulder_angle": shoulder_angle,
            "elbow_angle": elbow_angle
        }

    return jsonify(
        status_data
    )


@app.route(
    "/tracking/start",
    methods=["POST"]
)
def start_tracking():

    global tracking_enabled
    global current_command

    with state_lock:

        tracking_enabled = True
        current_command = (
            "TRACKING ENABLED"
        )

    return jsonify({
        "success": True,
        "tracking_enabled": True
    })


@app.route(
    "/tracking/pause",
    methods=["POST"]
)
def pause_tracking():

    global tracking_enabled
    global current_command

    with state_lock:

        tracking_enabled = False
        current_command = (
            "TRACKING PAUSED"
        )

    return jsonify({
        "success": True,
        "tracking_enabled": False
    })


@app.route(
    "/motors/mean",
    methods=["POST"]
)
def motors_mean():

    global tracking_enabled

    with state_lock:
        tracking_enabled = False

    mean_thread = threading.Thread(
        target=return_motors_to_mean,
        name="ReturnToMeanThread",
        daemon=True
    )

    mean_thread.start()

    return jsonify({
        "success": True,
        "message": (
            "Returning motors to mean"
        )
    })


@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "camera_running": camera_running,
        "model_running": model_running
    })


# ============================================================
# SIGNAL AND CLEANUP
# ============================================================

def handle_shutdown_signal(
    signal_number,
    frame
):

    print(
        "Shutdown signal received: "
        + str(signal_number)
    )

    shutdown_event.set()


def release_camera():

    global camera_running

    if camera is not None:

        try:

            camera.release()
            print("Camera released.")

        except Exception as error:

            print(
                "Camera release error: "
                + str(error)
            )

    camera_running = False


def cleanup_system():

    global model_running

    print("Cleaning up robot system...")

    try:

        return_motors_to_mean_during_shutdown()

    except Exception:

        print(
            "Error returning motors to mean:"
        )

        traceback.print_exc()

    release_camera()

    model_running = False

    print("Robot system cleanup complete.")


# ============================================================
# MAIN
# ============================================================

def main():

    global web_server

    signal.signal(
        signal.SIGINT,
        handle_shutdown_signal
    )

    signal.signal(
        signal.SIGTERM,
        handle_shutdown_signal
    )

    tracking_thread = None

    try:

        initialize_motors()
        initialize_model()
        initialize_camera()

        tracking_thread = threading.Thread(
            target=camera_tracking_loop,
            name="CameraTrackingThread",
            daemon=True
        )

        tracking_thread.start()

        hostname = socket.gethostname()

        print()
        print(
            "Robot arm tracking server is running."
        )
        print()
        print(
            "Open from another device:"
        )
        print(
            f"http://{hostname}.local:{WEB_PORT}"
        )
        print()

        web_server = make_server(
            WEB_HOST,
            WEB_PORT,
            app,
            threaded=True
        )

        web_server.timeout = 1.0

        while not shutdown_event.is_set():

            web_server.handle_request()

    except KeyboardInterrupt:

        shutdown_event.set()

    except Exception:

        print("Fatal robot program error:")
        traceback.print_exc()

        shutdown_event.set()

    finally:

        shutdown_event.set()

        if (
            tracking_thread is not None
            and tracking_thread.is_alive()
        ):

            tracking_thread.join(
                timeout=3.0
            )

        cleanup_system()


if __name__ == "__main__":
    main()
