# Raspberry Pi Person Tracking Camera

This project uses a Raspberry Pi 5, USB camera, OpenCV, and SSD MobileNet to detect and track people in a live camera feed.

## Goal

The goal is to build the vision system for a robot arm that can detect a person and decide whether the arm should move left, right, or stop based on the person's position in the camera frame.

## Current Features

- Displays live USB camera feed from `/dev/video0`
- Runs SSD MobileNet object detection using OpenCV DNN
- Filters detections to track only humans
- Calculates whether the detected person is left, right, or centered
- Displays FPS on the camera feed
- Prints movement commands:
  - `LEFT`
  - `RIGHT`
  - `STOP`
  - `NO PERSON`

## Hardware Used

- Raspberry Pi 5
- USB camera
- microSD card with Raspberry Pi OS
- Monitor or Raspberry Pi Connect screen sharing

## Software Used

- Python
- OpenCV
- SSD MobileNet
- Raspberry Pi OS
- Raspberry Pi Connect




source ssd-env/bin/activate
