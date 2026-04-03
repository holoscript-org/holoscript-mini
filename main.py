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

    # NEW — how long (seconds) to keep a nav-event label visible on-screen
    NAV_DISPLAY_DURATION = 0.6
    # NEW — pixels to pan per frame for each continuous nav event
    NAV_PAN_STEP         = 4.0
    # NEW — maximum pan offset in either axis (pixels)
    NAV_PAN_MAX          = 250.0

    def __init__(self):
        self._last_rotation_y: float = 0.0   # used when frozen

        # NEW — nav overlay state (display-only, does not affect engine)
        self._nav_display_event: str   = ""    # last event received
        self._nav_display_until: float = 0.0   # monotonic deadline

        # NEW — continuous pan offset accumulated from nav events
        self._nav_offset_x: float = 0.0
        self._nav_offset_y: float = 0.0

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

    def _to_screen(self, corners: np.ndarray, scale: float,
                   ox: float = 0.0, oy: float = 0.0) -> np.ndarray:
        """Scale + translate to screen centre (+ optional nav pan offset)."""
        cx = self.WIDTH  // 2 + ox
        cy = self.HEIGHT // 2 + oy
        half = self.BASE_HALF * scale
        pts = corners * half
        pts[:, 0] += cx
        pts[:, 1] += cy
        return pts.astype(np.int32).reshape((-1, 1, 2))

    # ------------------------------------------------------------------
    # Draw one frame
    # ------------------------------------------------------------------

    def _draw_frame(self, rotation_y: float, scale: float,
                    frozen: bool, gesture: str,
                    nav_ox: float = 0.0, nav_oy: float = 0.0) -> np.ndarray:
        canvas = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)

        # Draw subtle grid background
        grid_color = (20, 20, 20)
        for x in range(0, self.WIDTH, 40):
            cv2.line(canvas, (x, 0), (x, self.HEIGHT), grid_color, 1)
        for y in range(0, self.HEIGHT, 40):
            cv2.line(canvas, (0, y), (self.WIDTH, y), grid_color, 1)

        # Square colour changes with state + PSF phase
        nav_phase = scene_state.nav_phase   # NEW ─ read PSF phase once
        if frozen:
            color = (80, 80, 255)     # blue when frozen
        elif gesture == "ROTATE":
            color = (0, 220, 255)     # cyan-yellow when rotating
        elif gesture == "ZOOM":
            color = (0, 255, 100)     # green when zooming
        elif nav_phase == "FOCUS" or gesture == "FOCUS":   # NEW
            color = (0, 200, 255)     # gold in FOCUS
        elif nav_phase == "PREVIEW":                        # NEW
            color = (30, 160, 255)    # amber in PREVIEW
        elif gesture in ("POINT", "FOCUS"):
            color = (220, 80, 220)    # purple in navigation
        else:
            color = (200, 200, 200)   # default grey

        corners = self._square_corners()
        corners = self._rotate_corners(corners, rotation_y)
        pts = self._to_screen(corners, scale, nav_ox, nav_oy)

        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=3)
        cv2.fillPoly(canvas, [pts], color=tuple(c // 4 for c in color))  # dim fill

        # HUD
        if nav_phase == "FOCUS" or gesture == "FOCUS":
            status = "FOCUS"
        elif nav_phase == "PREVIEW":
            status = "PREVIEW"
        elif frozen:
            status = "FROZEN"
        else:
            status = gesture or "\u2014"
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

        # NEW ─ PREVIEW ring: pulsing amber circle around square centre
        if nav_phase == "PREVIEW":
            pulse_r = int(self.BASE_HALF * scale * 1.55 + 6 * math.sin(time.monotonic() * 8))
            ox_i, oy_i = int(nav_ox), int(nav_oy)
            cv2.circle(canvas,
                       (self.WIDTH // 2 + ox_i, self.HEIGHT // 2 + oy_i),
                       pulse_r, (30, 160, 255), 2)
            cv2.putText(canvas, "PREVIEW",
                        (self.WIDTH // 2 - 55, self.HEIGHT // 2 - self.BASE_HALF * 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 160, 255), 2)

        # NEW ─ FOCUS ring: solid gold thick ring
        elif nav_phase == "FOCUS" or gesture == "FOCUS":
            focus_r = int(self.BASE_HALF * scale * 1.65)
            ox_i, oy_i = int(nav_ox), int(nav_oy)
            cv2.circle(canvas,
                       (self.WIDTH // 2 + ox_i, self.HEIGHT // 2 + oy_i),
                       focus_r, (0, 200, 255), 4)
            cv2.putText(canvas, "[ FOCUS ]",
                        (self.WIDTH // 2 - 72, self.HEIGHT // 2 - self.BASE_HALF * 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 200, 255), 2)

        # NEW ─ nav event overlay (shown for NAV_DISPLAY_DURATION seconds)
        if self._nav_display_event and time.monotonic() < self._nav_display_until:
            label = self._nav_display_event
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 2.0, 3)
            tx = (self.WIDTH  - tw) // 2
            ty = (self.HEIGHT + th) // 2
            pad = 18
            cv2.rectangle(canvas,
                          (tx - pad, ty - th - pad),
                          (tx + tw + pad, ty + pad),
                          (30, 10, 30), cv2.FILLED)
            cv2.putText(canvas, label,
                        (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 2.0, (220, 80, 220), 3)
        elif gesture in ("POINT", "FOCUS") and nav_phase == "NAV":
            cv2.putText(canvas, "POINT \u2014 move to navigate",
                        (self.WIDTH // 2 - 160, self.HEIGHT // 2 - self.BASE_HALF - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 80, 180), 1)

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
        print("  POINT  + move hand -> navigate (NEXT/PREV/UP/DOWN)")
        print("  'q'                -> quit\n")

        target_fps   = 30
        frame_period = 1.0 / target_fps

        while True:
            t_start = time.time()

            rotation_y, scale, _explode, frozen = scene_state.get_render_params()
            gesture = scene_state.current_gesture

            # NEW — consume nav event; accumulate pan offset + latch display
            nav = scene_state.consume_nav_event()
            if nav:
                step = self.NAV_PAN_STEP
                if   nav == "NEXT": self._nav_offset_x += step
                elif nav == "PREV": self._nav_offset_x -= step
                elif nav == "DOWN": self._nav_offset_y += step
                elif nav == "UP":   self._nav_offset_y -= step
                cap = self.NAV_PAN_MAX
                self._nav_offset_x = max(-cap, min(cap, self._nav_offset_x))
                self._nav_offset_y = max(-cap, min(cap, self._nav_offset_y))
                self._nav_display_event = nav
                self._nav_display_until = time.monotonic() + self.NAV_DISPLAY_DURATION

            # Reset pan offset when leaving POINT mode
            if gesture != "POINT":
                self._nav_offset_x *= 0.85  # smooth drift-back
                self._nav_offset_y *= 0.85
                if abs(self._nav_offset_x) < 1.0: self._nav_offset_x = 0.0
                if abs(self._nav_offset_y) < 1.0: self._nav_offset_y = 0.0

            # When frozen, stop advancing the displayed rotation
            if frozen:
                rotation_y = self._last_rotation_y
            else:
                self._last_rotation_y = rotation_y

            frame = self._draw_frame(rotation_y, scale, frozen, gesture,
                                     self._nav_offset_x, self._nav_offset_y)
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
