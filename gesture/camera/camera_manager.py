"""Camera manager for video capture and frame processing.

This module provides functions to initialize, read from, and release the camera.
It handles basic error checking and provides debug output for frame capture.
"""

import cv2
from camera_config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, PRINT_INTERVAL


def initialize_camera():
    """Initialize the camera with configured settings.
    
    Opens the camera device, sets resolution, and verifies initialization.
    
    Returns:
        cv2.VideoCapture: Initialized camera capture object
        
    Raises:
        RuntimeError: If camera fails to open or resolution cannot be set
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera at index {CAMERA_INDEX}")
    
    # Set frame dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    # Verify resolution was set
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Camera initialized: {actual_width}x{actual_height}")
    
    return cap


def read_frame(cap, frame_counter):
    """Read a frame from the camera.
    
    Reads a frame, flips it horizontally for mirror effect, and provides
    periodic debug output.
    
    Args:
        cap: OpenCV VideoCapture object
        frame_counter: Current frame number for debug output
        
    Returns:
        numpy.ndarray: Flipped frame image
        
    Raises:
        RuntimeError: If frame read fails
    """
    ret, frame = cap.read()
    
    if not ret:
        raise RuntimeError("Failed to read frame from camera")
    
    # Flip horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Print frame info at intervals
    if frame_counter % PRINT_INTERVAL == 0:
        print(f"Frame {frame_counter}: shape = {frame.shape}")
    
    return frame


def release_camera(cap):
    """Release the camera and clean up resources.
    
    Properly closes the camera device and destroys all OpenCV windows.
    
    Args:
        cap: OpenCV VideoCapture object to release
    """
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed")
