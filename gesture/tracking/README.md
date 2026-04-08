# Gesture Detection

Real-time **hand gesture detection system** built using **MediaPipe** and **OpenCV**.
This project detects hand landmarks from a webcam and classifies gestures such as pinch, fist, open palm, point, and V sign.

---

## Features

* Real-time webcam gesture tracking
* MediaPipe hand landmark detection
* Gesture classification system
* Pinch detection
* Finger curl detection
* Hand motion tracking
* Thread-safe gesture engine

---

## Supported Gestures

| Gesture   | Description                      |
| --------- | -------------------------------- |
| OPEN_PALM | All fingers extended             |
| FIST      | All fingers closed               |
| PINCH     | Thumb and index finger together  |
| POINT     | Index finger extended            |
| V_SIGN    | Index and middle finger extended |

---

## Project Structure

```
gesture-detection
│
├── core
│   └── gesture
│       └── gesture_engine.py
│
├── gesture
│   ├── classification
│   │   ├── gesture_classifier.py
│   │   └── demo_live_gestures.py
│   │
│   └── tracking
│
├── tests
│
├── hand_landmarker.task
│
└── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/khushiiii4/gesture-detection.git
cd gesture-detection
```

Install dependencies:

```
pip install mediapipe opencv-python numpy
```

---

## Running the Demo

Run the live gesture detection demo:

```
python gesture/classification/demo_live_gestures.py
```

Press **Q** to exit the camera window.

---

## Requirements

* Python 3.9+
* OpenCV
* MediaPipe
* NumPy

---

## Future Improvements

* Gesture smoothing and filtering
* 3D hand pose tracking
* Integration with hologram renderer
* Voice + gesture interaction system

---

## Author

Khushi
