"""Quick threaded demo for new continuous pose tracking."""
import time, sys
sys.path.insert(0, 'c:/Users/khush/holoscript-mini')
from gesture.gesture_engine import GestureEngine

print("=== THREADED MODE + CONTINUOUS POSE DEMO (Day 2 v2 — full feature extraction) ===")
print("Show your hand to the camera. Running for 20 seconds...\n")

engine = GestureEngine()
engine.start_thread()

try:
    for i in range(20):
        time.sleep(1)
        pose = engine.get_hand_pose()
        if pose:
            cx, cy, cz = pose["center"]
            vx, vy, vz = pose["palm_direction"]
            print(
                f"[{i+1:02d}] gesture={engine.get_current_gesture()['gesture']:<10s} "
                f"pinch={pose['pinch_strength']:.2f}(act={pose['pinch_active']})  "
                f"curl={pose['avg_curl']:.2f}(fist={pose['fist_active']})  "
                f"dx={pose['dx']:+.4f} dy={pose['dy']:+.4f}  "
                f"dir=({vx:+.2f},{vy:+.2f},{vz:+.2f})"
            )
        else:
            print(f"[{i+1:02d}] no hand detected")
except KeyboardInterrupt:
    print("\nInterrupted by user")
finally:
    engine.stop()
    print("\nDemo complete.")
