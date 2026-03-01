"""Non-interactive camera runner: init, read one frame, print shape, then exit."""

import sys
from pathlib import Path

# Ensure local camera modules import correctly
sys.path.insert(0, str(Path(__file__).parent))

from camera_manager import initialize_camera, read_frame, release_camera


def main():
    print("Starting non-interactive camera run_once.py")

    try:
        cap = initialize_camera()
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return

    try:
        frame = read_frame(cap, 0)
        print("Got frame shape:", frame.shape)
    except Exception as e:
        print(f"Error reading frame: {e}")
    finally:
        release_camera(cap)


if __name__ == "__main__":
    main()
