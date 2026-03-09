"""
main.py — Week-4 Simple Renderer
=================================
Visualises SceneState transformations (rotation_y, scale, frozen)
driven by the GestureEngine.

Controls (via hand gestures):
    PINCH  + move  → rotate square
    OPEN HAND + move  → zoom square
    FIST           → freeze animation
    'q' key        → quit
"""

import sys
import math
import time
import numpy as np
import cv2
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from core.state.scene_state import scene_state
from gesture.gesture_engine import GestureEngine


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class SimpleRenderer:
    """OpenCV-based renderer that reflects SceneState every frame."""

    # Canvas size
    WIDTH  = 800
    HEIGHT = 600

    # Base half-size of the square (in pixels, before scale)
    BASE_HALF = 100

    def __init__(self):
        self._last_rotation_y: float = 0.0   # used when frozen

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _square_corners(self) -> np.ndarray:
        """Return 4 corners of the unit square (±1, ±1) as (4, 2)."""
        return np.array([
            [-1.0, -1.0],
            [ 1.0, -1.0],
            [ 1.0,  1.0],
            [-1.0,  1.0],
        ], dtype=np.float32)

    def _rotate_corners(self, corners: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate (4,2) corners around the origin by angle_deg (Y-axis maps to 2-D rotation)."""
        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rot = np.array([[cos_a, -sin_a],
                        [sin_a,  cos_a]], dtype=np.float32)
        return corners @ rot.T

    def _to_screen(self, corners: np.ndarray, scale: float) -> np.ndarray:
        """Scale + translate to screen centre. Returns integer pixel array (4,1,2)."""
        cx, cy = self.WIDTH // 2, self.HEIGHT // 2
        half = self.BASE_HALF * scale
        pts = corners * half
        pts[:, 0] += cx
        pts[:, 1] += cy
        return pts.astype(np.int32).reshape((-1, 1, 2))

    # ------------------------------------------------------------------
    # Draw one frame
    # ------------------------------------------------------------------

    def _draw_frame(self, rotation_y: float, scale: float,
                    frozen: bool, gesture: str) -> np.ndarray:
        canvas = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)

        # Draw subtle grid background
        grid_color = (20, 20, 20)
        for x in range(0, self.WIDTH, 40):
            cv2.line(canvas, (x, 0), (x, self.HEIGHT), grid_color, 1)
        for y in range(0, self.HEIGHT, 40):
            cv2.line(canvas, (0, y), (self.WIDTH, y), grid_color, 1)

        # Square colour changes with state
        if frozen:
            color = (80, 80, 255)     # blue-ish when frozen
        elif gesture == "ROTATE":
            color = (0, 220, 255)     # yellow when rotating
        elif gesture == "ZOOM":
            color = (0, 255, 100)     # green when zooming
        else:
            color = (200, 200, 200)   # default grey

        corners = self._square_corners()
        corners = self._rotate_corners(corners, rotation_y)
        pts = self._to_screen(corners, scale)

        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=3)
        cv2.fillPoly(canvas, [pts], color=tuple(c // 4 for c in color))  # dim fill

        # HUD
        status = "FROZEN" if frozen else gesture or "—"
        cv2.putText(canvas, f"Gesture : {status}",
                    (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"Rotation: {rotation_y % 360:.1f} deg",
                    (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
        cv2.putText(canvas, f"Scale   : {scale:.2f}x",
                    (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
        cv2.putText(canvas, "Press 'q' to quit",
                    (20, self.HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (100, 100, 100), 1)

        if frozen:
            cv2.putText(canvas, "[ FROZEN ]",
                        (self.WIDTH // 2 - 80, self.HEIGHT // 2 - self.BASE_HALF - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 255), 2)

        return canvas

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Render loop — reads SceneState every frame at ~30 FPS."""
        print("HoloScript Renderer started.")
        print("  PINCH  + move hand -> rotate")
        print("  OPEN HAND + move   -> zoom")
        print("  FIST               -> freeze")
        print("  'q'                -> quit\n")

        target_fps   = 30
        frame_period = 1.0 / target_fps

        while True:
            t_start = time.time()

            rotation_y, scale, _explode, frozen = scene_state.get_render_params()
            gesture = scene_state.current_gesture

            # When frozen, stop advancing the displayed rotation
            if frozen:
                rotation_y = self._last_rotation_y
            else:
                self._last_rotation_y = rotation_y

            frame = self._draw_frame(rotation_y, scale, frozen, gesture)
            cv2.imshow("HoloScript Renderer", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Maintain ~30 FPS
            elapsed = time.time() - t_start
            sleep_time = frame_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        cv2.destroyAllWindows()
        print("Renderer stopped.")


# ---------------------------------------------------------------------------
# Entry point — starts GestureEngine thread then renderer
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = GestureEngine()
    engine.start_thread()

    try:
        renderer = SimpleRenderer()
        renderer.run()
    finally:
        engine.stop()
