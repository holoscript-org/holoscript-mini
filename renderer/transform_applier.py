"""
renderer/transform_applier.py
==============================
Apply per-frame global and per-object OpenGL transforms driven by the
shared :class:`~renderer.scene_state.SceneState` blackboard.

Pipeline position
-----------------
::

    AnimationEngine.tick()  →  TransformApplier  →  OpenGL draw calls

Transform hierarchy applied each frame
----------------------------------------
.. code-block:: text

    glPushMatrix()
        glRotatef(rotation_y, 0, 1, 0)   ← user/gesture Y-rotation
        glScalef(scale, scale, scale)     ← user/gesture uniform scale
        for each object:
            glPushMatrix()
                glTranslatef(*exploded_world_pos)  ← object placement + explode
                <draw object at local origin>
            glPopMatrix()
    glPopMatrix()

Explode model
-------------
When ``explode > 0`` each object is pushed radially outward from the scene
origin by::

    direction = world_position / ‖world_position‖
    exploded_pos = world_position + direction × explode_factor

Objects sitting exactly at the origin (‖pos‖ < ε) receive no explode
displacement, avoiding a zero-division degenerate case.

Usage::

    applier = TransformApplier()

    # ----- in the render loop (called once per frame) -----
    # Read a consistent snapshot of all transform params from the blackboard:
    rotation_y, scale, explode, frozen = scene_state.get_render_params()

    glPushMatrix()
    applier.apply(scene_state)                     # rotation_y + scale

    for obj in objects:
        with applier.object_transform(obj, explode):
            _draw_object(obj)                      # draw at local origin

    glPopMatrix()

Notes
-----
* ``apply()`` **must** be called while a valid OpenGL context is active.
* This module does **not** modify any :class:`~renderer.scene_builder.RenderObject`
  field — all transforms live solely on the OpenGL matrix stack.
* The context managers handle ``glPushMatrix`` / ``glPopMatrix`` pairing
  automatically, guaranteeing stack balance even if drawing raises.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Generator, TYPE_CHECKING

import numpy as np

from OpenGL.GL import (
    glPushMatrix,
    glPopMatrix,
    glRotatef,
    glScalef,
    glTranslatef,
)

from renderer.scene_builder import RenderObject

if TYPE_CHECKING:
    from renderer.scene_state import SceneState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPSILON: float = 1e-10   # minimum ‖pos‖ before explode offset is suppressed


# ---------------------------------------------------------------------------
# TransformApplier
# ---------------------------------------------------------------------------


class TransformApplier:
    """Applies scene-level and per-object OpenGL transforms each render frame.

    The class is **stateless**: all transform parameters are read from
    :class:`~renderer.scene_state.SceneState` on every call, so it is safe
    to reuse a single instance across multiple frames and multiple scenes.
    """

    # ------------------------------------------------------------------
    # Global scene transforms
    # ------------------------------------------------------------------

    def apply(self, scene_state: "SceneState") -> None:
        """Push ``rotation_y`` and ``scale`` onto the current OpenGL matrix.

        Reads a consistent atomic snapshot of transform parameters from
        *scene_state* via :meth:`~renderer.scene_state.SceneState.get_render_params`
        (single lock acquisition) and calls:

        1. ``glRotatef(rotation_y, 0, 1, 0)`` — Y-axis rotation.
        2. ``glScalef(scale, scale, scale)`` — uniform scale.

        The caller is responsible for surrounding calls with
        ``glPushMatrix`` / ``glPopMatrix`` to avoid polluting the matrix stack.

        Args:
            scene_state: Live blackboard; only ``rotation_y`` and ``scale``
                are consumed here.  ``explode`` is handled per-object via
                :meth:`apply_object_transform` or :meth:`object_transform`.
        """
        rotation_y, scale, _explode, _frozen = scene_state.get_render_params()
        glRotatef(float(rotation_y), 0.0, 1.0, 0.0)
        glScalef(float(scale), float(scale), float(scale))

    # ------------------------------------------------------------------
    # Per-object explode transform
    # ------------------------------------------------------------------

    def apply_object_transform(
        self,
        obj: RenderObject,
        explode: float,
    ) -> None:
        """Translate to the object's (optionally exploded) world position.

        Computes the final render position as::

            if ‖world_position‖ > ε:
                direction = world_position / ‖world_position‖
                render_pos = world_position + direction × explode
            else:
                render_pos = world_position

        Then calls ``glTranslatef(*render_pos)``.

        The object's :attr:`~renderer.scene_builder.RenderObject.world_position`
        is **never** modified — the offset exists only on the OpenGL matrix stack.

        Args:
            obj:     The object to position.  Only ``world_position`` is read.
            explode: Radial outward displacement in world units.  ``0.0``
                     means no displacement; valid range is ``[0.0, 1.0]``
                     as enforced by :class:`~renderer.scene_state.SceneState`
                     but any float is accepted here for composability.
        """
        pos: np.ndarray = obj.world_position   # shape (3,), float64 — do not mutate
        norm: float = float(np.linalg.norm(pos))

        if norm > _EPSILON and explode != 0.0:
            direction: np.ndarray = pos / norm
            render_pos: np.ndarray = pos + direction * float(explode)
        else:
            render_pos = pos

        glTranslatef(float(render_pos[0]), float(render_pos[1]), float(render_pos[2]))

    # ------------------------------------------------------------------
    # Context managers for matrix-stack safety
    # ------------------------------------------------------------------

    @contextmanager
    def scene_transform(
        self,
        scene_state: "SceneState",
    ) -> Generator[None, None, None]:
        """Context manager that applies and cleans up the global scene transform.

        Wraps :meth:`apply` in a ``glPushMatrix`` / ``glPopMatrix`` pair,
        guaranteeing stack balance even if an exception is raised inside the
        ``with`` block.

        Usage::

            with applier.scene_transform(scene_state):
                for obj in objects:
                    with applier.object_transform(obj, explode):
                        _draw_object(obj)

        Args:
            scene_state: Live blackboard passed directly to :meth:`apply`.

        Yields:
            Nothing; the ``with`` block body executes between push and pop.
        """
        glPushMatrix()
        try:
            self.apply(scene_state)
            yield
        finally:
            glPopMatrix()

    @contextmanager
    def object_transform(
        self,
        obj: RenderObject,
        explode: float,
    ) -> Generator[None, None, None]:
        """Context manager that positions a single object and cleans up.

        Wraps :meth:`apply_object_transform` in a ``glPushMatrix`` /
        ``glPopMatrix`` pair.

        Usage::

            _, _, explode, _ = scene_state.get_render_params()
            with applier.object_transform(obj, explode):
                _draw_object(obj)   # draws at local origin after translation

        Args:
            obj:     Object to translate; see :meth:`apply_object_transform`.
            explode: Explode factor; see :meth:`apply_object_transform`.

        Yields:
            Nothing; the ``with`` block body executes between push and pop.
        """
        glPushMatrix()
        try:
            self.apply_object_transform(obj, explode)
            yield
        finally:
            glPopMatrix()

    # ------------------------------------------------------------------
    # Static helper
    # ------------------------------------------------------------------

    @staticmethod
    def compute_explode_offset(
        world_position: np.ndarray,
        explode: float,
    ) -> np.ndarray:
        """Return the explode displacement vector for *world_position*.

        Can be used by downstream code that needs the exploded position as a
        NumPy array without touching the OpenGL matrix stack (e.g. for
        CPU-side bounding-box queries or label positioning).

        Args:
            world_position: Object centre in world space, shape ``(3,)``.
            explode:        Radial displacement magnitude in world units.

        Returns:
            Shape-``(3,)`` float64 offset vector.  When the object is at the
            origin or *explode* is zero the zero vector is returned.
        """
        pos = np.asarray(world_position, dtype=np.float64)
        norm = float(np.linalg.norm(pos))
        if norm > _EPSILON and explode != 0.0:
            return (pos / norm) * float(explode)
        return np.zeros(3, dtype=np.float64)
