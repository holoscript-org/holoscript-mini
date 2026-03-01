# Gesture Recognition Module

This module provides gesture recognition capabilities for the holoscript-mini project.

## Structure

### camera/
Manages camera input and video capture.
- `camera_manager.py`: Video capture management
- `camera_config.py`: Camera configuration settings

### tracking/
Hand detection and tracking functionality.
- `hand_tracker.py`: Hand landmark tracking
- `tracking_config.py`: Tracking configuration

### processing/
Processes hand landmark data.
- `landmark_processor.py`: Landmark data processing
- `geometry_utils.py`: Geometric calculations
- `smoothing_utils.py`: Data smoothing algorithms

### calibration/
User calibration and profile management.
- `calibration_manager.py`: Calibration process management
- `calibration_profile.json`: Default calibration profile

### classification/ (EMPTY THIS WEEK)
Gesture classification and recognition.
- `gesture_classifier.py`: Gesture classification logic
- `debounce_manager.py`: Gesture debouncing

### debug/
Debugging and visualization tools.
- `debug_logger.py`: Debug logging
- `visual_overlay.py`: Visual debugging overlays

### test/
Unit tests for the gesture system.
- `test_gesture_engine.py`: Test suite

## Usage

```python
from gesture.camera import CameraManager
from gesture.tracking import HandTracker

# Initialize components
camera = CameraManager()
tracker = HandTracker()

# Start processing
camera.start()
frame = camera.get_frame()
landmarks = tracker.process_frame(frame)
```

## Dependencies

- OpenCV (cv2)
- MediaPipe (or similar hand tracking library)
- NumPy

## Status

Current implementation includes basic structure. Classification module will be completed in future iterations.
