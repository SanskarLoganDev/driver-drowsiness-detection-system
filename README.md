# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a driver drowsiness detection system that uses three non-invasive methods to detect drowsiness:
1. **Eye Aspect Ratio (EAR) monitoring** - Detects drowsiness through eye closure patterns
2. **Yawning detection** - Measures mouth opening distance between lips
3. **Heart rate monitoring** - Arduino-based pulse sensor with cloud monitoring via ThingSpeak

## Development Environment Setup

### Python Dependencies
```bash
pip install -r requirements.txt
```

Required packages: scipy, Pillow, imutils, numpy, pygame, opencv-python, opencv-contrib-python, cmake, dlib

### Required Assets
The `asset/` directory must contain:
- `alarm.mp3` - Audio alert file played when drowsiness is detected
- `shape_predictor_68_face_landmarks.dat` - dlib's pre-trained facial landmark detector (99MB file)

Note: The shape predictor file is required for facial landmark detection and can be downloaded from dlib's model repository.

### Hardware Setup (Optional)
For pulse monitoring: Arduino IDE with ESP8266WiFi library, connected to NodeMCU with pulse sensor on pin A0.

## Running the Detection Systems

### Eye Drowsiness Detection
```bash
python src/eye-drowsiness.py
```
- Opens a tkinter GUI window (720x740, non-resizable)
- Requires webcam access (VideoCapture index 0)
- Press the "Exit!" button or close window to stop
- Default thresholds: EYE_THRESHOLD=0.20, EYE_FRAMES=15

### Yawning Detection
```bash
python src/yawning-detection.py
```
- Opens two OpenCV windows: "Live Landmarks" and "Yawn Detection"
- Requires webcam access (VideoCapture index 0)
- Press Enter key to exit
- Default threshold: lip_distance > 25 triggers yawn detection

### Heart Rate Monitoring
Upload `hardware/heart-rate-monitoring.ino` to NodeMCU using Arduino IDE:
- Configure WiFi credentials in the code (ssid, password)
- Set ThingSpeak channel number and API key
- Pulse sensor connected to A0 pin
- Data uploads every 400ms to ThingSpeak

## Architecture

### Eye Drowsiness Detection (src/eye-drowsiness.py)
- Uses dlib's HOG-based face detector and 68-point facial landmark predictor
- Calculates Eye Aspect Ratio (EAR) using Euclidean distances between eye landmarks
- Left eye: landmarks[lStart:lEnd], Right eye: landmarks[rStart:rEnd]
- Triggers alarm via pygame mixer when EAR < threshold for consecutive frames
- Main loop: `video_loop()` runs every 1ms, processes frame through `drowsy_detection()`
- GUI built with tkinter, displays real-time video feed with annotations

### Yawning Detection (src/yawning-detection.py)
- Uses same dlib models for facial landmark detection
- Focuses on mouth landmarks (indices 50-67 for lips)
- Calculates vertical distance between top_lip (mean of points 50-52, 61-63) and bottom_lip (mean of points 56-58, 65-67)
- Tracks yawn state transitions to count discrete yawns (prevents counting a single yawn multiple times)
- Displays annotated landmarks with point indices for debugging

### Hardware Integration (hardware/heart-rate-monitoring.ino)
- Analog pulse sensor reading scaled to 0-100 range
- WiFi connectivity via ESP8266WiFi library
- ThingSpeak integration for cloud-based monitoring and graphing
- Designed to detect drowsiness through pulse rate changes (inverse correlation)

## Key Implementation Details

### Facial Landmark System
Both Python scripts use dlib's 68-point facial landmark model:
- Points 36-41: Left eye
- Points 42-47: Right eye
- Points 48-67: Mouth region (inner and outer contours)

### Threshold Tuning
- Eye drowsiness: Adjust `EYE_THRESHOLD` (lower = more sensitive) and `EYE_FRAMES` (higher = requires longer closure)
- Yawning: Adjust `lip_distance > 25` comparison value in yawning-detection.py:74

### Camera Configuration
Both scripts use `cv2.VideoCapture(0)` for default webcam. Change index if using external camera.

## Known Dependencies and Constraints

- Virtual environment exists at `.venv/` (not tracked in git)
- Python 3.x required (tested on 3.11.3)
- Windows-compatible paths used throughout
- Hardware code requires Arduino IDE with ESP8266 support; will show errors in other IDEs
- dlib installation may require Visual Studio C++ build tools on Windows
