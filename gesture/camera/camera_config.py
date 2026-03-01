"""Camera configuration settings.

This module defines constants for camera initialization and frame capture.
These settings control video input parameters for the gesture recognition system.

Constants:
    CAMERA_INDEX: Index of the camera device to use (0 for default camera)
    FRAME_WIDTH: Width of captured frames in pixels
    FRAME_HEIGHT: Height of captured frames in pixels
    PRINT_INTERVAL: Number of frames between debug print statements
"""

# Camera device index (0 = default camera)
CAMERA_INDEX = 0

# Frame dimensions
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Debug print interval (frames)
PRINT_INTERVAL = 60
