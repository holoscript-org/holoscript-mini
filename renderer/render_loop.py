"""
renderer/render_loop.py
=======================
Renderer thread: drives the OpenGL rendering pipeline at 25 FPS using
Pyglet + PyOpenGL, reading from and writing to the shared
:class:`~renderer.scene_state.SceneState` blackboard.

Pipeline position
-----------------
::

    SceneState
        ↓  scene_json / rotation_y / scale / explode / frozen
    Renderer.render_scene()
        ├─ detect_scene_change() → SceneBuilder.parse_and_build()
        ├─ AnimationEngine.tick()
        ├─ TransformApplier (scene + per-object transforms)
        ├─ _draw_object() per primitive
        ├─ FrameExtractor.extract() → raw_frame (WINDOW_H × WINDOW_W × 3)
        ├─ cylindrical_engine.build_frame() → cylindrical frame (360 × 18 × 3)
        └─ SceneState.current_frame ← (360, 18, 3) uint8

Threading model
---------------
* :meth:`Renderer.start` spawns a **daemon thread** named ``RendererThread``.
* That thread creates the Pyglet window, activates the GL context, and runs
  the manual 25-FPS loop until :meth:`Renderer.stop` is called.
* **No other thread touches the window or GL context.**

Frame capture
-------------
:meth:`_capture_and_write_frame` reads the back-buffer with ``glReadPixels``
via :class:`~renderer.frame_extractor.FrameExtractor` into a
``(WINDOW_H, WINDOW_W, 3) uint8`` NumPy array stored in :attr:`raw_frame`.

The raw frame is then converted to the ``(360, 18, 3)`` cylindrical LED
display format by :func:`~cylindrical_engine.frame_builder.build_frame`,
which applies scene-level transforms (rotation_y, scale) and anti-aliased
cylindrical projection before writing to
:attr:`~renderer.scene_state.SceneState.current_frame`.

Supported primitives
--------------------
+----------+-----------------------------------+
| type     | implementation                    |
+==========+===================================+
| sphere   | ``gluSphere`` (GLU quadric)       |
+----------+-----------------------------------+
| cube     | ``GL_QUADS`` (6 hand-coded faces) |
+----------+-----------------------------------+
| cylinder | ``gluCylinder`` + disk caps       |
+----------+-----------------------------------+
| ring     | torus triangle-strip tessellation |
+----------+-----------------------------------+
| label    | ``pyglet.text.Label`` (2-D HUD)   |
+----------+-----------------------------------+
"""

from __future__ import annotations

import math
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyglet
import pyglet.text
from pyglet.gl import Config as GLConfig

from OpenGL.GL import (
    GL_AMBIENT,
    GL_AMBIENT_AND_DIFFUSE,
    GL_BACK,
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_COLOR_MATERIAL,
    GL_CULL_FACE,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DIFFUSE,
    GL_EMISSION,
    GL_FLOAT,
    GL_FRONT,
    GL_FRONT_AND_BACK,
    GL_LIGHT0,
    GL_LIGHTING,
    GL_LINE_LOOP,
    GL_MODELVIEW,
    GL_NORMALIZE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_PROJECTION,
    GL_QUADS,
    GL_RGB,
    GL_SHADE_MODEL,
    GL_SMOOTH,
    GL_SRC_ALPHA,
    GL_TRIANGLE_STRIP,
    GL_UNSIGNED_BYTE,
    GLfloat,
    glBegin,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor3f,
    glColorMaterial,
    glDisable,
    glEnable,
    glEnd,
    glFrustum,
    glLightfv,
    glLoadIdentity,
    glMaterialfv,
    glMatrixMode,
    glNormal3f,
    glPopMatrix,
    glPushMatrix,
    glReadPixels,
    glRotatef,
    glScalef,
    glShadeModel,
    glTranslatef,
    glVertex3f,
    glViewport,
)
from OpenGL.GLU import (
    GLU_SMOOTH,
    gluCylinder,
    gluDeleteQuadric,
    gluDisk,
    gluLookAt,
    gluNewQuadric,
    gluPerspective,
    gluQuadricNormals,
    gluSphere,
)

from renderer.animation import AnimationEngine
from renderer.frame_extractor import FrameExtractor
from renderer.scene_builder import RenderObject, SceneBuildError, SceneBuilder
from renderer.scene_state import SceneState
from renderer.transform_applier import TransformApplier
from cylindrical_engine.frame_builder import build_frame


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_FPS: int = 25
_FRAME_DT: float = 1.0 / TARGET_FPS
WINDOW_W: int = 800
WINDOW_H: int = 800

# Torus tessellation quality (ring primitive)
_TORUS_MAJOR_SEGS: int = 64   # segments around the major circle
_TORUS_MINOR_SEGS: int = 16   # segments around the minor circle

# Default camera position (world space)
_CAM_EYE: Tuple[float, float, float] = (0.0, 6.0, 18.0)
_CAM_AT:  Tuple[float, float, float] = (0.0, 0.0, 0.0)
_CAM_UP:  Tuple[float, float, float] = (0.0, 1.0, 0.0)

# Sentinel distinguishing "no scene loaded yet" from `None` intentionally.
_SENTINEL: object = object()

# Material colours for emissive objects
_EMISSIVE_STRENGTH: float = 0.4
_ZERO4 = (GLfloat * 4)(0.0, 0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class Renderer:
    """Owns the Pyglet window and drives the 25-FPS OpenGL render loop.

    Lifecycle::

        renderer = Renderer(scene_state)
        renderer.start()   # spawns RendererThread
        ...
        renderer.stop()    # signals thread to exit

    The window created by this class is not visible by default.  Pass
    ``visible=True`` to the constructor for a debugging preview.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        scene_state: SceneState,
        *,
        visible: bool = False,
    ) -> None:
        """Initialise the renderer.

        Args:
            scene_state: Shared blackboard.  Only the renderer thread writes
                ``current_frame``; all other state is read.
            visible:     If ``True`` the Pyglet window is shown on screen
                (useful for development).  Defaults to ``False`` (offscreen).
        """
        self._state: SceneState = scene_state
        self._visible: bool = visible

        # Threading
        self._thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()

        # Pipeline objects (created inside run_loop on the renderer thread)
        self._window: Optional[pyglet.window.Window] = None
        self._quadric: Optional[object] = None  # GLU quadric handle
        self._labels: Dict[str, pyglet.text.Label] = {}

        # Scene graph
        self._builder: SceneBuilder = SceneBuilder()
        self._anim: AnimationEngine = AnimationEngine()
        self._transform: TransformApplier = TransformApplier()
        self._frame_extractor: FrameExtractor = FrameExtractor()
        self._objects: List[RenderObject] = []

        # Change detection
        self._last_scene_json: object = _SENTINEL
        self._bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)

        # Raw framebuffer snapshot (WINDOW_H × WINDOW_W × 3, uint8)
        self._raw_frame: Optional[np.ndarray] = None
        self._raw_frame_lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the renderer thread.

        Safe to call exactly once.  Subsequent calls are no-ops.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self.run_loop,
            name="RendererThread",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the renderer thread to exit and join it (with a 3-second timeout)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    @property
    def raw_frame(self) -> Optional[np.ndarray]:
        """Latest raw framebuffer capture as ``(WINDOW_H, WINDOW_W, 3) uint8``.

        Thread-safe read.  Returns ``None`` until the first frame is rendered.
        This property is consumed by the cylindrical projection engine.
        """
        with self._raw_frame_lock:
            return self._raw_frame

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def run_loop(self) -> None:
        """Create the Pyglet window and run the manual 25-FPS render loop.

        This method runs entirely on the renderer thread.  It exits when
        :meth:`stop` sets the stop event.
        """
        try:
            self._window = self._create_window()
            self._window.switch_to()
            self._setup_gl()
            self._quadric = gluNewQuadric()
            gluQuadricNormals(self._quadric, GLU_SMOOTH)
            self._state.append_log("RendererThread: OpenGL context ready")

            while not self._stop_event.is_set():
                t0 = time.monotonic()

                self._window.switch_to()
                self._window.dispatch_events()

                if self._window.has_exit:  # user closed the window
                    break

                self.render_scene(_FRAME_DT)
                self._window.flip()

                # Maintain target FPS
                elapsed = time.monotonic() - t0
                sleep_time = _FRAME_DT - elapsed
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

        except Exception as exc:  # pragma: no cover — runtime GL errors
            self._state.append_log(f"RendererThread fatal error: {exc}")
            raise
        finally:
            if self._quadric is not None:
                try:
                    gluDeleteQuadric(self._quadric)
                except Exception:
                    pass
            if self._window is not None:
                try:
                    self._window.close()
                except Exception:
                    pass
            self._state.append_log("RendererThread: stopped")

    # ------------------------------------------------------------------
    # Scene change detection
    # ------------------------------------------------------------------

    def detect_scene_change(self) -> bool:
        """Return ``True`` if ``scene_json`` has changed since the last call.

        Uses identity comparison (``is not``) so that a new dict object
        triggers a rebuild even if its contents are identical.  This matches
        the contract in :class:`~renderer.scene_state.SceneState` where the
        setter always replaces the reference.

        Returns:
            ``True`` on the first call and whenever the reference changes.
        """
        current = self._state.scene_json
        if current is not self._last_scene_json:
            self._last_scene_json = current
            return True
        return False

    # ------------------------------------------------------------------
    # Frame orchestration
    # ------------------------------------------------------------------

    def render_scene(self, dt: float) -> None:
        """Execute one complete render frame.

        Frame workflow:

        1. Read ``scene_json`` from :attr:`_state`.
        2. Detect scene change → rebuild object graph via :class:`SceneBuilder`.
        3. Feed ``dt`` to :class:`AnimationEngine` for orbital animation.
        4. Clear the framebuffer and set up the camera.
        5. Apply global transforms (rotation_y, scale) via :class:`TransformApplier`.
        6. Draw each object in the scene graph.
        7. Draw label overlays (2-D HUD pass).
        8. Capture the framebuffer with ``glReadPixels`` → write ``current_frame``.

        Args:
            dt: Frame delta time in seconds.  Typically ``1/25``.
        """
        # ── 1 & 2: scene change ──────────────────────────────────────
        if self.detect_scene_change():
            self._rebuild_scene()

        # ── 3: animation ─────────────────────────────────────────────
        self._anim.tick(self._objects, dt, self._state)

        # ── 4: clear + camera ────────────────────────────────────────
        r, g, b = self._bg_color
        glClearColor(r, g, b, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            _CAM_EYE[0], _CAM_EYE[1], _CAM_EYE[2],
            _CAM_AT[0],  _CAM_AT[1],  _CAM_AT[2],
            _CAM_UP[0],  _CAM_UP[1],  _CAM_UP[2],
        )

        # Update the light position after the camera is placed so it stays
        # fixed in world space.
        _set_light_position(10.0, 10.0, 10.0)

        # ── 5 & 6: scene transforms + draw ───────────────────────────
        _, _, explode, _ = self._state.get_render_params()
        glEnable(GL_LIGHTING)

        with self._transform.scene_transform(self._state):
            for obj in self._objects:
                if obj.type == "label":
                    continue  # labels drawn in a separate 2-D pass
                with self._transform.object_transform(obj, explode):
                    self._draw_object(obj)

        # ── 7: label overlay ─────────────────────────────────────────
        self._draw_labels(explode)

        # ── 7 & 8: frame extraction + cylindrical conversion ───────
        self._capture_and_write_frame()

    # ------------------------------------------------------------------
    # Scene rebuild
    # ------------------------------------------------------------------

    def _rebuild_scene(self) -> None:
        """(Re)build the object graph from the current ``scene_json``."""
        scene_json = self._state.scene_json

        # Clear stale label widgets
        self._labels.clear()

        if scene_json is None:
            self._objects = []
            self._anim.reset()
            self._state.append_log("RendererThread: scene cleared")
            return

        try:
            self._objects = self._builder.parse_and_build(scene_json)
            self._anim.reset()

            # Extract background colour (optional field)
            raw_bg = scene_json.get("background_color", "#000000")
            try:
                self._bg_color = _parse_hex_color(raw_bg)
            except ValueError:
                self._bg_color = (0.0, 0.0, 0.0)

            self._state.append_log(
                f"RendererThread: built scene with {len(self._objects)} object(s)"
            )
        except SceneBuildError as exc:
            self._objects = []
            self._state.append_log(f"RendererThread SceneBuildError: {exc}")

    # ------------------------------------------------------------------
    # Object drawing dispatch
    # ------------------------------------------------------------------

    def _draw_object(self, obj: RenderObject) -> None:
        """Dispatch to the correct primitive draw routine for *obj*.

        Emissive objects have a self-illumination material set; non-emissive
        objects use the standard diffuse/ambient lighting.

        Args:
            obj: A :class:`~renderer.scene_builder.RenderObject` whose
                ``world_position`` has already been applied to the matrix
                stack by the caller via :class:`TransformApplier`.  Drawing
                occurs at the local origin.
        """
        r, g, b = obj.color
        glColor3f(r, g, b)

        if obj.emissive:
            emissive = (GLfloat * 4)(r * _EMISSIVE_STRENGTH,
                                     g * _EMISSIVE_STRENGTH,
                                     b * _EMISSIVE_STRENGTH, 1.0)
            glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, emissive)
        else:
            glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, _ZERO4)

        dispatch = {
            "sphere":   self._draw_sphere,
            "cube":     self._draw_cube,
            "cylinder": self._draw_cylinder,
            "ring":     self._draw_ring,
        }
        draw_fn = dispatch.get(obj.type)
        if draw_fn is not None:
            draw_fn(obj.size)

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    def _draw_sphere(self, size: float) -> None:
        """Draw a smooth sphere of radius *size* at the local origin.

        Uses the GLU quadric object for tessellated normals (needed for
        correct specular/diffuse lighting).

        Args:
            size: Radius in world units.
        """
        gluSphere(self._quadric, size, 32, 32)

    def _draw_cube(self, size: float) -> None:
        """Draw a solid axis-aligned cube with side length ``2 × size``.

        Each face is drawn as a single ``GL_QUADS`` primitive with a correct
        outward face normal.

        Args:
            size: Half-extent (distance from centre to face).
        """
        s = size
        # Vertices and normals for the 6 faces: (normal, [4 vertices])
        faces = [
            # +X
            ((1, 0, 0), [( s,-s,-s),( s, s,-s),( s, s, s),( s,-s, s)]),
            # -X
            ((-1, 0, 0), [(-s,-s, s),(-s, s, s),(-s, s,-s),(-s,-s,-s)]),
            # +Y
            ((0, 1, 0), [(-s, s,-s),( s, s,-s),( s, s, s),(-s, s, s)]),
            # -Y
            ((0,-1, 0), [(-s,-s, s),( s,-s, s),( s,-s,-s),(-s,-s,-s)]),
            # +Z
            ((0, 0, 1), [(-s,-s, s),( s,-s, s),( s, s, s),(-s, s, s)]),
            # -Z
            ((0, 0,-1), [( s,-s,-s),(-s,-s,-s),(-s, s,-s),( s, s,-s)]),
        ]
        glBegin(GL_QUADS)
        for normal, verts in faces:
            glNormal3f(*normal)
            for v in verts:
                glVertex3f(*v)
        glEnd()

    def _draw_cylinder(self, size: float) -> None:
        """Draw a capped cylinder of radius ``size/2`` and height ``size``.

        The cylinder axis is aligned to +Y.  Both end-caps are closed with
        GLU disks.

        Args:
            size: Controls both the radius (``size/2``) and height (``size``).
        """
        radius = size * 0.5
        height = size

        # Side surface
        gluCylinder(self._quadric, radius, radius, height, 32, 1)

        # Bottom cap (facing -Y → rotate 180° around X to flip normal)
        glPushMatrix()
        glRotatef(180.0, 1.0, 0.0, 0.0)
        gluDisk(self._quadric, 0.0, radius, 32, 1)
        glPopMatrix()

        # Top cap (translate to top then draw facing +Y)
        glPushMatrix()
        glTranslatef(0.0, 0.0, height)
        gluDisk(self._quadric, 0.0, radius, 32, 1)
        glPopMatrix()

    def _draw_ring(self, size: float) -> None:
        """Draw a torus (ring) in the XZ plane using triangle strips.

        The torus is defined by a major radius ``size`` (distance from the
        torus centre to the tube centre) and a minor radius ``size * 0.25``
        (tube cross-section radius).

        Tessellation: :data:`_TORUS_MAJOR_SEGS` × :data:`_TORUS_MINOR_SEGS`.

        Args:
            size: Major radius of the torus in world units.
        """
        R = size          # major radius
        r = size * 0.25   # minor (tube) radius
        M = _TORUS_MAJOR_SEGS
        N = _TORUS_MINOR_SEGS

        for i in range(M):
            phi0 = (2.0 * math.pi * i)       / M
            phi1 = (2.0 * math.pi * (i + 1)) / M
            cos_phi0, sin_phi0 = math.cos(phi0), math.sin(phi0)
            cos_phi1, sin_phi1 = math.cos(phi1), math.sin(phi1)

            glBegin(GL_TRIANGLE_STRIP)
            for j in range(N + 1):
                theta = (2.0 * math.pi * j) / N
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)

                # Vertex on phi1 strip
                x1 = (R + r * cos_t) * cos_phi1
                y1 = r * sin_t
                z1 = (R + r * cos_t) * sin_phi1
                nx1 = cos_t * cos_phi1
                ny1 = sin_t
                nz1 = cos_t * sin_phi1
                glNormal3f(nx1, ny1, nz1)
                glVertex3f(x1, y1, z1)

                # Vertex on phi0 strip
                x0 = (R + r * cos_t) * cos_phi0
                y0 = r * sin_t
                z0 = (R + r * cos_t) * sin_phi0
                nx0 = cos_t * cos_phi0
                ny0 = sin_t
                nz0 = cos_t * sin_phi0
                glNormal3f(nx0, ny0, nz0)
                glVertex3f(x0, y0, z0)
            glEnd()

    # ------------------------------------------------------------------
    # Label rendering (2-D HUD pass)
    # ------------------------------------------------------------------

    def _draw_labels(self, explode: float) -> None:
        """Render label objects as 2-D screen-space text (HUD pass).

        Each label object in the scene is drawn as a ``pyglet.text.Label``
        positioned at the 2-D projection of its (optionally exploded)
        world position.  The pass runs *after* the 3-D scene draw, with
        lighting disabled, so labels always appear on top.

        Labels are cached in :attr:`_labels` by object ``id`` and reused
        across frames; the cache is cleared when the scene is rebuilt.

        Args:
            explode: Current explode factor from SceneState.
        """
        label_objects = [o for o in self._objects if o.type == "label"]
        if not label_objects:
            return

        # For each label, project its 3-D world position to 2-D window coords.
        rotation_y, scale, _exp, _frz = self._state.get_render_params()

        glDisable(GL_LIGHTING)

        for obj in label_objects:
            # Compute explode offset
            offset = TransformApplier.compute_explode_offset(obj.world_position, explode)
            world_pos = obj.world_position + offset

            screen_x, screen_y = _project_to_screen(
                world_pos,
                rotation_y,
                scale,
                WINDOW_W,
                WINDOW_H,
            )

            # Skip labels that project behind the camera
            if screen_x is None:
                continue

            # Create or reuse the pyglet label
            label = self._labels.get(obj.id)
            text = obj.id
            r, g, b = obj.color
            color_bytes = (
                int(r * 255), int(g * 255), int(b * 255), 255
            )

            if label is None:
                label = pyglet.text.Label(
                    text,
                    font_name="Arial",
                    font_size=10,
                    x=int(screen_x),
                    y=int(screen_y),
                    anchor_x="center",
                    anchor_y="center",
                    color=color_bytes,
                )
                self._labels[obj.id] = label
            else:
                label.x = int(screen_x)
                label.y = int(screen_y)
                label.color = color_bytes

            label.draw()

        glEnable(GL_LIGHTING)

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def _capture_and_write_frame(self) -> None:
        """Extract the back-buffer and write the cylindrical frame to SceneState.

        Full pipeline:

        1. :class:`~renderer.frame_extractor.FrameExtractor` calls
           ``glReadPixels`` directly into a reused NumPy buffer, reshapes and
           flips it → ``(WINDOW_H, WINDOW_W, 3) uint8``.
        2. The raw frame is stored in :attr:`_raw_frame` under the lock for
           external consumers (e.g. debugging / recording).
        3. :func:`~cylindrical_engine.frame_builder.build_frame` converts the
           animated object graph — together with the current scene-state
           transforms — into the ``(360, 18, 3) uint8`` cylindrical LED frame.
        4. The result is written atomically to
           :attr:`~renderer.scene_state.SceneState.current_frame`.
        """
        # ── Step 1: glReadPixels → (WINDOW_H, WINDOW_W, 3) uint8 ─────────
        raw = self._frame_extractor.extract(WINDOW_W, WINDOW_H)

        # ── Step 2: store raw frame (thread-safe) ────────────────────────
        with self._raw_frame_lock:
            self._raw_frame = raw

        # ── Step 3: cylindrical projection ──────────────────────────────
        # build_frame() reads rotation_y and scale from scene_state internally
        # via SceneState.get_render_params(), then projects object world
        # positions through the cylindrical mapping with anti-aliasing and
        # brighter-wins overlap resolution.
        cyl_frame = build_frame(self._objects, self._state)   # (360, 18, 3) uint8

        # ── Step 4: write to blackboard ──────────────────────────────────
        self._state.current_frame = cyl_frame

    # ------------------------------------------------------------------
    # OpenGL initialisation
    # ------------------------------------------------------------------

    def _setup_gl(self) -> None:
        """One-time OpenGL state setup called after the context is active."""
        glViewport(0, 0, WINDOW_W, WINDOW_H)

        # Projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, WINDOW_W / WINDOW_H, 0.1, 200.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Depth test
        glEnable(GL_DEPTH_TEST)

        # Smooth shading
        glShadeModel(GL_SMOOTH)

        # Lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

        ambient  = (GLfloat * 4)(0.15, 0.15, 0.15, 1.0)
        diffuse  = (GLfloat * 4)(0.9,  0.9,  0.9,  1.0)
        specular = (GLfloat * 4)(0.5,  0.5,  0.5,  1.0)
        glLightfv(GL_LIGHT0, GL_AMBIENT,  ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  diffuse)

        # Auto-normalise normals after glScalef
        glEnable(GL_NORMALIZE)

        # Let glColor3f drive the material diffuse + ambient
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # Blending (used by labels)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # ------------------------------------------------------------------
    # Window factory
    # ------------------------------------------------------------------

    def _create_window(self) -> pyglet.window.Window:
        """Create and return a Pyglet window with a depth-buffered GL config.

        Returns:
            A :class:`pyglet.window.Window` whose GL context is compatible
            with PyOpenGL calls.
        """
        try:
            config = GLConfig(
                major_version=2,
                minor_version=1,
                double_buffer=True,
                depth_size=24,
            )
            window = pyglet.window.Window(
                width=WINDOW_W,
                height=WINDOW_H,
                caption="HoloScript",
                visible=self._visible,
                config=config,
            )
        except Exception:
            # Fallback: default config (e.g. on headless CI)
            window = pyglet.window.Window(
                width=WINDOW_W,
                height=WINDOW_H,
                caption="HoloScript",
                visible=self._visible,
            )

        @window.event
        def on_resize(width: int, height: int) -> None:  # noqa: unused
            glViewport(0, 0, width, height)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45.0, max(width, 1) / max(height, 1), 0.1, 200.0)
            glMatrixMode(GL_MODELVIEW)

        return window


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _set_light_position(x: float, y: float, z: float) -> None:
    """Set ``GL_LIGHT0`` to a positional light at ``(x, y, z)``."""
    pos = (GLfloat * 4)(x, y, z, 1.0)
    glLightfv(GL_LIGHT0, 1, pos)   # 1 == GL_POSITION (portable constant)


def _parse_hex_color(hex_str: str) -> Tuple[float, float, float]:
    """Convert ``#RRGGBB`` to ``(r, g, b)`` floats in ``[0, 1]``.

    Args:
        hex_str: Colour string, e.g. ``"#1A2B3C"``.

    Returns:
        ``(r, g, b)`` each in ``[0.0, 1.0]``.

    Raises:
        ValueError: If the string is not a valid ``#RRGGBB`` hex colour.
    """
    import re
    match = re.fullmatch(r"#([0-9A-Fa-f]{6})", hex_str)
    if not match:
        raise ValueError(f"Invalid hex colour: {hex_str!r}")
    h = match.group(1)
    return int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0


def _project_to_screen(
    world_pos: np.ndarray,
    rotation_y: float,
    scale: float,
    width: int,
    height: int,
) -> Tuple[Optional[float], Optional[float]]:
    """Project a world-space point to window pixel coordinates.

    Applies the same ``rotation_y`` + ``scale`` transform used by the
    :class:`TransformApplier` scene transform, then the camera view, then
    perspective division.

    Args:
        world_pos:  3-D world position, shape ``(3,)``.
        rotation_y: Y-axis rotation in degrees.
        scale:      Uniform scale factor.
        width:      Window width in pixels.
        height:     Window height in pixels.

    Returns:
        ``(screen_x, screen_y)`` in window pixels, or ``(None, None)`` for
        points behind the near plane.
    """
    # Apply scene transform (rotation_y around Y, then scale)
    rad_y = math.radians(rotation_y)
    cy, sy = math.cos(rad_y), math.sin(rad_y)

    x, y, z = float(world_pos[0]), float(world_pos[1]), float(world_pos[2])

    # Y-rotation
    xr =  x * cy + z * sy
    yr =  y
    zr = -x * sy + z * cy

    # Scale
    xr *= scale
    yr *= scale
    zr *= scale

    # Camera view: translate by -eye then the look-at orientation.
    # Use the inverse of the gluLookAt camera placed at _CAM_EYE looking at origin.
    ex, ey, ez = _CAM_EYE
    # forward = normalize(_CAM_AT - _CAM_EYE)
    fx, fy, fz = -ex, -ey, -ez
    fn = math.hypot(math.hypot(fx, fy), fz) or 1.0
    fx, fy, fz = fx / fn, fy / fn, fz / fn

    # right = forward × up
    ux, uy, uz = _CAM_UP
    rx = fy * uz - fz * uy
    ry = fz * ux - fx * uz
    rz = fx * uy - fy * ux
    rn = math.hypot(math.hypot(rx, ry), rz) or 1.0
    rx, ry, rz = rx / rn, ry / rn, rz / rn

    # up = right × forward
    ux2 = ry * fz - rz * fy
    uy2 = rz * fx - rx * fz
    uz2 = rx * fy - ry * fx

    # Translate
    tx = xr - ex
    ty = yr - ey
    tz = zr - ez

    # View space
    vx = rx * tx + ry * ty + rz * tz
    vy = ux2 * tx + uy2 * ty + uz2 * tz
    vz = fx * tx + fy * ty + fz * tz

    # vz is the depth along the forward axis (>0 = in front of camera).
    # Perspective division uses vz directly; negative/zero means behind camera.
    if vz <= 0.1:
        return None, None

    fov_rad = math.radians(45.0)
    f = 1.0 / math.tan(fov_rad * 0.5)
    aspect = width / height

    px = (vx / vz) * f / aspect
    py = (vy / vz) * f

    # NDC [-1,1] to window pixels
    sx = (px + 1.0) * 0.5 * width
    sy = (py + 1.0) * 0.5 * height

    return sx, sy
