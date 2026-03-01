"""Demo script for the Gesture Engine.

This script demonstrates both blocking and threaded modes of the GestureEngine.
"""

import time
import sys
from pathlib import Path

# Add gesture module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gesture.gesture_engine import GestureEngine


def demo_blocking_mode():
    """Demo the blocking mode (DAY 1 style)."""
    print("=== BLOCKING MODE DEMO ===")
    print("Starting gesture engine in blocking mode...")
    print("Press 'q' in the camera window to quit")
    
    engine = GestureEngine()
    engine.start()  # This blocks until 'q' is pressed


def demo_threaded_mode():
    """Demo the threaded mode (DAY 4 style)."""
    print("\n=== THREADED MODE DEMO ===")
    print("Starting gesture engine in background thread...")
    
    engine = GestureEngine()
    engine.start_thread()
    
    try:
        # Simulate doing other work while gesture detection runs
        for i in range(20):
            time.sleep(1)
            
            # Get current gesture
            gesture_data = engine.get_current_gesture()
            print(f"[{i+1}] Current gesture: {gesture_data['gesture']} "
                  f"(confidence: {gesture_data['confidence']:.2f})")
            
            # Demo state update (DAY 3 feature)
            class MockState:
                def __init__(self):
                    self.current_gesture = None
                    self.gesture_confidence = 0.0
            
            mock_state = MockState()
            engine.update_state(mock_state)
            
            print(f"     State updated - gesture: {mock_state.current_gesture}, "
                  f"confidence: {mock_state.gesture_confidence:.2f}")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted")
    
    finally:
        engine.stop()
        print("Demo completed")


def main():
    """Main demo function."""
    print("Gesture Engine Demo")
    print("===================")
    print("This demo shows the complete gesture engine implementation")
    print("with all DAY 1-4 requirements:")
    print("- MediaPipe hand detection")
    print("- 5-gesture classification (OPEN_PALM, FIST, PINCH, POINT, V_SIGN)")
    print("- Debounce and confidence scoring")
    print("- State update preparation")
    print("- Thread-safe operation")
    print()
    
    print("Choose demo mode:")
    print("1. Blocking mode (press 'q' to quit)")
    print("2. Threaded mode (20-second auto demo)")
    print("3. Both modes")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        demo_blocking_mode()
    elif choice == '2':
        demo_threaded_mode()
    elif choice == '3':
        demo_blocking_mode()
        demo_threaded_mode()
    else:
        print("Invalid choice. Running threaded demo...")
        demo_threaded_mode()


if __name__ == "__main__":
    main()