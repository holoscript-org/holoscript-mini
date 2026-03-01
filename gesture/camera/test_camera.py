"""Test script for camera manager.

This script tests the camera manager by:
- Initializing the camera
- Continuously reading and displaying frames
- Accepting keyboard input to exit ('q' key)
- Properly releasing camera resources

No MediaPipe or gesture processing - just basic camera functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import cv2
from camera_manager import initialize_camera, read_frame, release_camera


def main():
    """Run camera test loop."""
    print("Starting camera test...")
    print("Press 'q' to quit")
    
    # Initialize camera
    try:
        cap = initialize_camera()
    except RuntimeError as e:
        print(f"Error: {e}")
        return
    
    frame_counter = 0
    
    try:
        # Main loop
        while True:
            # Read frame
            try:
                frame = read_frame(cap, frame_counter)
                frame_counter += 1
            except RuntimeError as e:
                print(f"Error reading frame: {e}")
                break
            
            # Display frame
            cv2.imshow("Camera Test", frame)
            
            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"\nExiting after {frame_counter} frames")
                break
    
    except KeyboardInterrupt:
        print(f"\nInterrupted after {frame_counter} frames")
    
    finally:
        # Clean up
        release_camera(cap)
        print("Test completed")


if __name__ == "__main__":
    main()
