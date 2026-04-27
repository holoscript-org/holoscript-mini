# -*- coding: utf-8 -*-
# Visual confirmation checklist (fill in after manual test):
# [ ] Window opens at 800x800
# [ ] Background is near-black (dark blue-black)
# [ ] Sun is visible at center, gold colored, no shading (emissive)
# [ ] Earth is visible to the right of sun, blue, shaded
# [ ] Red cube is visible above center, shaded
# [ ] Green cylinder is visible to the left, shaded
# [ ] Yellow ring is visible in front, flat disk shape
# [ ] FPS printed to terminal every 25 frames, value >= 20
# [ ] No crashes after 30 seconds of running
#
# Phase 4 Visual Checks:
# [ ] Window title shows "HoloScript Phase 4 - Static Scene"
# [ ] All 9 planets visible (sun + 8 planets in a line)
# [ ] Sun is gold colored, largest, unshaded (emissive)
# [ ] Mercury is small grey sphere closest to sun
# [ ] Earth is blue, correct position
# [ ] Jupiter and Saturn are noticeably larger than inner planets
# [ ] Neptune and Uranus visible at far end
# [ ] No objects clipped or missing at screen edges
# [ ] FPS still >= 20
# [ ] No crashes after 30 seconds
#
# Phase 5 Visual Checks (fixed in Phase 5.1 patch):
# [x] Window title shows "HoloScript Phase 5 - Animation"
# [x] All 8 planets are visibly moving (not static)
# [x] Mercury moves noticeably faster than Saturn
# [x] Sun stays completely stationary at center
# [x] Orbits are circular (not drifting off screen)  ← camera pulled back to (9,35,75), FOV=60
# [x] Press SPACE: all planets freeze instantly
# [x] Press SPACE again: all planets resume from exact position where they froze
# [x] FPS still >= 20 during animation               ← capped at ~25 via dt<1/30 guard
# [x] No crashes after 60 seconds of running
# Note: saturn=0.13, uranus=0.09, neptune=0.06 so outer planets spread apart
#
# Phase 6 Visual Checks:
# [ ] FPS is smooth and high (>= 60), no jitter
# [ ] Window title shows "HoloScript Phase 6 - Transforms"
# [ ] Planets still orbit correctly from Phase 5
# [ ] Press RIGHT arrow: entire scene rotates right smoothly
# [ ] Press LEFT arrow: entire scene rotates left smoothly
# [ ] Press UP arrow: scene zooms in (gets bigger)
# [ ] Press DOWN arrow: scene zooms out (gets smaller)
# [ ] Press E key multiple times: planets spread apart, sun stays at center
# [ ] Press R key: scene snaps back to default view
# [ ] Press SPACE: animation freezes, transforms still work while frozen
# [ ] No crashes after 60 seconds
#
# Phase 7 Visual Checks:
# [ ] Terminal prints "[FrameExtractor] Test frame saved to renderer/assets/test_frame.png" on startup
# [ ] test_frame.png exists and can be opened
# [ ] test_frame.png shows the solar system scene — planets, correct colors, correct orientation
# [ ] Image is NOT upside down (planets in middle vertically, not flipped)
# [ ] Image background is black
# [ ] Press S key: terminal prints "[Manual] Frame saved via S key", file overwritten
# [ ] Animation and transforms still work normally
# [ ] FPS not significantly degraded by extraction (>= 25)
# [ ] No crashes after 60 seconds
#
# Phase 8 Visual Checks:
# [ ] Window title shows "HoloScript Phase 8 - Cylindrical Engine"
# [ ] Terminal confirms scene_state.current_frame set every frame (no ValueError)
# [ ] Press V: terminal prints "[POV] Frame visualization saved to renderer/assets/pov_frame.png"
# [ ] pov_frame.png shows colored patches/bands -- NOT all black
# [ ] Sun (gold) appears as a strong band across many angles in Red+Green channels
# [ ] Planets appear as smaller colored dots at various angles
# [ ] Press V at different moments: dot positions change as planets orbit
# [ ] Animation in main window still works normally
# [ ] FPS not degraded below 20
# [ ] No crashes after 60 seconds
#
# Phase 10 Visual Checks:
# [ ] Window title shows "HoloScript Phase 10 - Full Integration"
# [ ] Solar system loads and animates on startup
# [ ] Press J: scene switches instantly to 3-object scene, terminal prints "[Renderer] Scene rebuilt: 3 objects"
# [ ] After pressing J: red planet orbits faster than blue
# [ ] Press K: solar system reloads, terminal prints "[Renderer] Scene rebuilt: 9 objects"
# [ ] Press J then K rapidly 5 times: no crash, scene always ends up correct
# [ ] scene_state.logs fills with FPS entries (confirmed via logs)
# [ ] All previous keys still work: SPACE, arrows, E, R, S, V
# [ ] FPS >= 25 during normal operation
# [ ] No crashes after 120 seconds

# Realistic Visual Checks:
# [ ] Sun has visible glow/corona around it,
#     not just a flat sphere
# [ ] Earth is blue with visible green continent
#     patches and white polar regions
# [ ] Mars is orange-red with white polar cap
#     visible at top and bottom
# [ ] Jupiter has clearly visible horizontal
#     brown/tan/cream bands across its surface
# [ ] Saturn has clearly visible horizontal bands
#     AND a flat ring system around its equator
# [ ] Saturn's rings are tilted at an angle
#     (not perfectly flat to camera)
# [ ] Uranus is distinctly ice-blue-green,
#     with subtle banding
# [ ] Neptune is deeper more saturated blue
#     than Uranus, with subtle banding
# [ ] Mercury is small grey sphere
# [ ] Venus is featureless yellow-amber sphere
# [ ] All planets still orbit correctly
# [ ] SPACE freeze still works
# [ ] FPS still >= 20 with all the extra geometry

"""renderer/render_window.py
Pyglet window and render loop for the HoloScript Renderer — Phase 10 (Final Integration).
Loads solar system JSON, animates orbits, applies global transforms
(rotation_y, scale, explode), extracts framebuffer to numpy each frame.

GL calls go through PyOpenGL; pyglet is used for window + event loop only.
A legacy (compatibility profile) context is requested so the fixed-function
pipeline (GL_LIGHTING, gluPerspective, etc.) is available.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pyglet
import pyglet.gl as _pgl          # for context config only
from PIL import Image

from OpenGL.GL import (
    glEnable, glDisable,
    GL_DEPTH_TEST, GL_LIGHTING, GL_LIGHT0, GL_COLOR_MATERIAL,
    GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE,
    GL_LIGHT_MODEL_AMBIENT,
    glColorMaterial, glClearColor, glClear,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    glMatrixMode, glLoadIdentity,
    GL_PROJECTION, GL_MODELVIEW,
    glLightfv, GL_POSITION, GL_DIFFUSE, GL_AMBIENT,
    GLfloat,
)
from OpenGL.GLU import gluPerspective, gluLookAt

# Allow `python renderer/render_window.py` to resolve top-level package imports.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RENDERER_ASSETS = os.path.join(_PROJECT_ROOT, "renderer", "assets")
_SOLAR_SYSTEM_PATH = os.path.join(_RENDERER_ASSETS, "solar_system.json")
_HUMAN_HEART_PATH  = os.path.join(_RENDERER_ASSETS, "human_heart.json")
if __package__ in (None, ""):
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

import renderer.engine.primitives as primitives
from renderer.scene_parser import SceneObject, parse_scene
from renderer.loader.scene_loader import load_scene_from_file
from renderer.engine.animation import Animator
from renderer.engine.transforms import TransformApplier
from renderer.frame_extractor import FrameExtractor
from renderer.engine.cylindrical.frame_builder import build_frame_from_scene
from renderer.scene_state_ref import scene_state
from core.state.ipc_store import publish_renderer_snapshot, read_scene_command, read_control_command


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obj(
    obj_id: str,
    obj_type: str,
    position: list,
    color: tuple,
    size: float,
    emissive: bool,
) -> SceneObject:
    pos = np.array(position, dtype=float)
    return SceneObject(
        id=obj_id,
        type=obj_type,
        position=pos,
        color=color,
        emissive=emissive,
        is_orbiting=False,
        orbit_center=np.array([0.0, 0.0, 0.0]),
        orbit_radius=0.0,
        orbit_speed=0.0,
        size=size,
        world_position=pos.copy(),
    )


# ---------------------------------------------------------------------------
# Default camera (used when a scene JSON has no "camera" block)
# ---------------------------------------------------------------------------

_DEFAULT_CAMERA = {
    "fov":    65.0,
    "near":   0.1,
    "far":    5000.0,
    "eye":    [13.5, 28.0, 65.0],
    "target": [13.5,  0.0,  0.0],
    "up":     [0.0,   1.0,  0.0],
}


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class Renderer:
    def __init__(self, scene_path: str = _SOLAR_SYSTEM_PATH) -> None:
        # Request a legacy (compatibility) context so fixed-function GL works
        config = _pgl.Config(
            double_buffer=True,
            depth_size=24,
            major_version=2,
            minor_version=1,
        )
        self.window = pyglet.window.Window(
            width=800, height=800,
            caption=f"HoloScript — {os.path.basename(scene_path)}",
            config=config,
        )

        self._setup_gl()

        self.scene_objects: list[SceneObject] = load_scene_from_file(scene_path)
        if not self.scene_objects:
            print("[Renderer] WARNING: scene loaded empty — falling back to gold sphere at origin.")
            self.scene_objects = [
                _make_obj("sun", "sphere", [0, 0, 0], (1.0, 0.84, 0.0), 2.0, True)
            ]

        self.animator = Animator()
        self.transform_applier = TransformApplier()
        self.frame_extractor = FrameExtractor(self.window.width, self.window.height)
        self._last_raw_frame = None
        self._last_scene_version: int = -1
        self._last_command_id: int = -1
        self._last_control_id: int = -1
        self._needs_frame_refresh: bool = True
        self._pov_interval_sec: float = 0.14
        self._last_pov_update: float = 0.0
        self._preview_path: str = os.path.join(_PROJECT_ROOT, ".runtime", "render_preview.jpg")
        self._preview_interval_sec: float = 0.07   # ~14 fps writes to disk
        self._last_preview_write: float = 0.0
        self._snapshot_interval_sec: float = 0.35
        self._last_snapshot_publish: float = 0.0
        os.makedirs(os.path.dirname(self._preview_path), exist_ok=True)
        self._frame_times: list = []
        self.last_frame_time: float = time.perf_counter()
        self._camera_config: dict = dict(_DEFAULT_CAMERA)

        self.frame_count: int = 0
        self.fps_log: list[float] = []
        self._dt_buf: list[float] = []
        self.last_time: float = time.perf_counter()

        # Ignore stale command files from previous runs so old actions
        # (e.g. a prior "h") are not replayed on startup.
        last_scene_cmd = read_scene_command()
        if last_scene_cmd is not None:
            self._last_command_id = int(last_scene_cmd["id"])
        last_ctrl_cmd = read_control_command()
        if last_ctrl_cmd is not None:
            self._last_control_id = int(last_ctrl_cmd["id"])

        # push_handlers registers both on_draw and on_key_press at full GPU speed
        self.window.push_handlers(self)

    # ------------------------------------------------------------------
    # Scene rebuild
    # ------------------------------------------------------------------

    def _rebuild_scene(self, scene_json: dict) -> None:
        """Parse new JSON, replace scene_objects, reset animation clock, update camera."""
        new_objects = parse_scene(scene_json)
        if not new_objects:
            print("[Renderer] WARNING: parse_scene returned empty list — keeping existing scene.")
            return
        self.scene_objects = new_objects
        self._needs_frame_refresh = True
        self.animator.reset()
        cam_data = scene_json.get("camera")
        if cam_data and isinstance(cam_data, dict):
            self._camera_config = {**_DEFAULT_CAMERA, **cam_data}
        else:
            self._camera_config = dict(_DEFAULT_CAMERA)
        msg = f"[Renderer] Scene rebuilt: {len(self.scene_objects)} objects"
        scene_state.append_log(msg)
        print(msg)

    # ------------------------------------------------------------------
    # GL setup (called once after context is current)
    # ------------------------------------------------------------------

    def _setup_gl(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glClearColor(0.0, 0.0, 0.05, 1.0)

        glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat * 4)(10.0, 20.0, 10.0, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  (GLfloat * 4)(1.0,  1.0,  1.0,  1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT,  (GLfloat * 4)(0.2,  0.2,  0.2,  1.0))

    # ------------------------------------------------------------------
    # Camera (called every frame)
    # ------------------------------------------------------------------

    def _setup_camera(self) -> None:
        cam = self._camera_config
        w, h = self.window.width, self.window.height
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(float(cam["fov"]), w / h, float(cam["near"]), float(cam["far"]))
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        eye = cam["eye"]
        tgt = cam["target"]
        up  = cam["up"]
        gluLookAt(
            float(eye[0]), float(eye[1]), float(eye[2]),
            float(tgt[0]), float(tgt[1]), float(tgt[2]),
            float(up[0]),  float(up[1]),  float(up[2]),
        )

    def _save_preview_image(self, raw_frame: np.ndarray) -> None:
        """Write a compact renderer preview image atomically for FastAPI consumers.

        Kept small (480 px) and low-quality (70) so the write finishes fast
        enough not to stall the render thread at 14 fps.
        """
        try:
            resampling = getattr(Image, "Resampling", Image)
            preview = Image.fromarray(raw_frame).resize(
                (480, 480),
                resample=resampling.NEAREST,   # faster than BILINEAR for live preview
            )
            tmp_path = self._preview_path + ".tmp.jpg"
            preview.save(tmp_path, format="JPEG", quality=70, optimize=False)
            os.replace(tmp_path, self._preview_path)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def on_key_press(self, symbol, _modifiers) -> None:
        key = pyglet.window.key
        if symbol == key.SPACE:
            scene_state.frozen = not scene_state.frozen
            state = "FROZEN" if scene_state.frozen else "RUNNING"
            print(f"[Animation] {state}")
        elif symbol == key.LEFT:
            scene_state.rotation_y -= 5.0
            print(f"rotation_y: {scene_state.rotation_y:.1f}")
        elif symbol == key.RIGHT:
            scene_state.rotation_y += 5.0
            print(f"rotation_y: {scene_state.rotation_y:.1f}")
        elif symbol == key.UP:
            scene_state.scale = min(3.0, scene_state.scale + 0.1)
            print(f"scale: {scene_state.scale:.2f}")
        elif symbol == key.DOWN:
            scene_state.scale = max(0.2, scene_state.scale - 0.1)
            print(f"scale: {scene_state.scale:.2f}")
        elif symbol == key.E:
            scene_state.explode = min(1.0, scene_state.explode + 0.1)
            print(f"explode: {scene_state.explode:.2f}")
        elif symbol == key.R:
            scene_state.explode = 0.0
            scene_state.rotation_y = 0.0
            scene_state.scale = 1.0
            print("[Transforms] Reset to defaults")
        elif symbol == key.S:
            self.frame_extractor._saved_test_frame = False
            frame = self.frame_extractor.extract()
            self._last_raw_frame = frame
            self.frame_extractor.save_test_frame(frame)
            print("[Manual] Frame saved via S key")
        elif symbol == key.J:
            test_scene = {
                "objects": [
                    {"id": "sun",      "type": "sphere", "position": [0.0, 0.0, 0.0],
                     "color": [1.0, 0.84, 0.0], "animation": "none",
                     "orbit_center": [0.0, 0.0, 0.0], "orbit_speed": 0.0},
                    {"id": "planet_a", "type": "sphere", "position": [5.0, 0.0, 0.0],
                     "color": [1.0, 0.0, 0.0], "animation": "orbit",
                     "orbit_center": [0.0, 0.0, 0.0], "orbit_speed": 1.0},
                    {"id": "planet_b", "type": "sphere", "position": [9.0, 0.0, 0.0],
                     "color": [0.0, 0.0, 1.0], "animation": "orbit",
                     "orbit_center": [0.0, 0.0, 0.0], "orbit_speed": 0.4},
                ]
            }
            scene_state.scene_json = test_scene
            print("[J key] Injected 3-object test scene into SceneState")
        elif symbol == key.K:
            with open(_SOLAR_SYSTEM_PATH, "r", encoding="utf-8") as _f:
                data = json.load(_f)
            scene_state.scene_json = data
            scene_state.rotation_y = 0.0
            scene_state.scale = 1.0
            scene_state.explode = 0.0
            print("[K key] Reloaded solar system from JSON file")
        elif symbol == key.H:
            with open(_HUMAN_HEART_PATH, "r", encoding="utf-8") as _f:
                data = json.load(_f)
            scene_state.scene_json = data
            scene_state.rotation_y = 0.0
            scene_state.scale = 1.0
            scene_state.explode = 0.0
            print("[H key] Human heart scene loaded")
        elif symbol == key.V:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            pov = scene_state.current_frame
            if pov is not None:
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                axes[0].imshow(pov[:, :, 0], cmap="Reds",
                               aspect="auto", vmin=0, vmax=255)
                axes[0].set_title("Red channel")
                axes[0].set_xlabel("LED index (0-17)")
                axes[0].set_ylabel("Angle index (0-359)")
                axes[1].imshow(pov[:, :, 1], cmap="Greens",
                               aspect="auto", vmin=0, vmax=255)
                axes[1].set_title("Green channel")
                axes[2].imshow(pov[:, :, 2], cmap="Blues",
                               aspect="auto", vmin=0, vmax=255)
                axes[2].set_title("Blue channel")
                plt.tight_layout()
                path = os.path.join(_RENDERER_ASSETS, "pov_frame.png")
                plt.savefig(path, dpi=150)
                plt.close()
                print(f"[POV] Frame visualization saved to {path}")

    def _apply_control(self, action: str) -> None:
        """Apply a single GUI keyboard action to this process's scene_state."""
        if action == "space":
            scene_state.frozen = not scene_state.frozen
        elif action == "left":
            scene_state.rotation_y -= 5.0
        elif action == "right":
            scene_state.rotation_y += 5.0
        elif action == "up":
            scene_state.scale = min(3.0, scene_state.scale + 0.1)
        elif action == "down":
            scene_state.scale = max(0.2, scene_state.scale - 0.1)
        elif action == "e":
            scene_state.explode = min(1.0, scene_state.explode + 0.1)
        elif action == "r":
            scene_state.explode    = 0.0
            scene_state.rotation_y = 0.0
            scene_state.scale      = 1.0
        elif action == "j":
            test_scene = {
                "objects": [
                    {"id": "sun",      "type": "sphere", "position": [0.0, 0.0, 0.0],
                     "color": [1.0, 0.84, 0.0], "animation": "none",
                     "orbit_center": [0.0, 0.0, 0.0], "orbit_speed": 0.0},
                    {"id": "planet_a", "type": "sphere", "position": [5.0, 0.0, 0.0],
                     "color": [1.0, 0.0, 0.0], "animation": "orbit",
                     "orbit_center": [0.0, 0.0, 0.0], "orbit_speed": 1.0},
                    {"id": "planet_b", "type": "sphere", "position": [9.0, 0.0, 0.0],
                     "color": [0.0, 0.0, 1.0], "animation": "orbit",
                     "orbit_center": [0.0, 0.0, 0.0], "orbit_speed": 0.4},
                ]
            }
            scene_state.scene_json = test_scene
        elif action == "k":
            with open(_SOLAR_SYSTEM_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            scene_state.scene_json  = data
            scene_state.rotation_y  = 0.0
            scene_state.scale       = 1.0
            scene_state.explode     = 0.0
        elif action == "h":
            with open(_HUMAN_HEART_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            scene_state.scene_json  = data
            scene_state.rotation_y  = 0.0
            scene_state.scale       = 1.0
            scene_state.explode     = 0.0

    def on_draw(self) -> None:
        command = read_scene_command()
        if command and command["id"] != self._last_command_id:
            scene_state.scene_json = command["scene"]
            self._last_command_id = command["id"]
            scene_state.append_log("[Renderer] Applied scene command from backend")

        ctrl = read_control_command()
        if ctrl and ctrl["id"] != self._last_control_id:
            self._apply_control(ctrl["action"])
            self._last_control_id = ctrl["id"]

        # Atomic read of scene_json + version to detect changes
        with scene_state._lock:
            scene_json = scene_state._scene_json
            version    = scene_state._scene_version
        if version != self._last_scene_version:
            if scene_json is not None:
                self._rebuild_scene(scene_json)
            self._last_scene_version = version

        # One atomic read — rotation_y/scale/explode/frozen all consistent
        rotation_y, scale, explode, frozen = scene_state.get_render_params()

        # Advance animation clock (skips update when frozen)
        now = time.perf_counter()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        self._frame_times.append(dt)
        self.animator.update(self.scene_objects, dt, frozen)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._setup_camera()

        # Apply global scale + rotation to the entire scene
        self.transform_applier.apply_global(rotation_y, scale)

        label_queue = []
        for obj in self.scene_objects:
            # Temporarily shift world_position by explode offset for this draw only
            offset = self.transform_applier.get_explode_offset(obj.world_position, explode)
            original_pos = obj.world_position.copy()
            obj.world_position = original_pos + offset
            primitives.dispatch(obj)
            # Collect label at the exploded position while it's still set
            if obj.label:
                y_off = obj.scale if obj.type == "mesh" else obj.size
                label_queue.append((obj.label, obj.world_position.copy(), y_off))
            obj.world_position = original_pos  # restore — animation state must not change

        primitives.draw_labels_2d(label_queue, self.window.width, self.window.height)

        # FPS tracking
        now = time.perf_counter()
        dt = now - self.last_time
        self.last_time = now
        self._dt_buf.append(dt)
        self.frame_count += 1

        if self.frame_count % 25 == 0:
            total = sum(self._dt_buf[-25:])
            fps = 25.0 / total if total > 0 else 0.0
            self.fps_log.append(fps)
            print(f"FPS: {fps:.1f}")

        # --- Frame extraction + cylindrical POV ---
        # For static scenes (heart), avoid expensive per-frame extraction/build.
        # Dynamic/orbiting scenes keep full-rate extraction for POV outputs.
        has_orbits = any(obj.is_orbiting for obj in self.scene_objects)
        frame_updated = False
        now = time.perf_counter()
        should_force_refresh = self._needs_frame_refresh
        should_update_pov = should_force_refresh or (
            has_orbits and (now - self._last_pov_update >= self._pov_interval_sec)
        )
        should_update_preview = should_force_refresh or (
            has_orbits and (now - self._last_preview_write >= self._preview_interval_sec)
        )

        if should_update_pov or should_update_preview:
            raw_frame = self.frame_extractor.extract()
            self.frame_extractor.save_test_frame(raw_frame)
            self._last_raw_frame = raw_frame

            if should_update_preview:
                self._save_preview_image(raw_frame)
                self._last_preview_write = now

            if should_update_pov:
                pov_frame = build_frame_from_scene(self.scene_objects)
                scene_state.current_frame = pov_frame
                frame_updated = True
                self._last_pov_update = now

            if should_force_refresh:
                self._needs_frame_refresh = False

        should_publish_snapshot = (now - self._last_snapshot_publish) >= self._snapshot_interval_sec
        if should_publish_snapshot:
            try:
                publish_renderer_snapshot(
                    scene=scene_state.scene_json,
                    logs=scene_state.logs,
                    status={
                        "rotation_y": rotation_y,
                        "scale": scale,
                        "explode": explode,
                        "frozen": frozen,
                        "gesture": scene_state.current_gesture,
                        "transcript": scene_state.transcript,
                    },
                    frame=scene_state.current_frame,
                    frame_updated=frame_updated,
                )
                self._last_snapshot_publish = now
            except Exception:
                # IPC failure should not crash live rendering.
                pass

        if self.frame_count % 25 == 0 and self._frame_times:
            avg_dt = sum(self._frame_times[-25:]) / min(25, len(self._frame_times))
            fps = 1.0 / avg_dt if avg_dt > 0 else 0.0
            scene_state.append_log(
                f"[Renderer] FPS={fps:.1f} objects={len(self.scene_objects)}")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        pyglet.app.run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HoloScript renderer")
    parser.add_argument(
        "--scene", "-s",
        default=_SOLAR_SYSTEM_PATH,
        help="Path to scene JSON file (default: solar_system.json)"
    )
    args = parser.parse_args()
    renderer = Renderer(scene_path=args.scene)
    renderer.run()
