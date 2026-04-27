"""renderer/transform_applier.py
Global transform helpers for the HoloScript Renderer.
Pure math + OpenGL calls. No pyglet. No SceneState.
"""

from __future__ import annotations

import numpy as np
from OpenGL.GL import glScalef, glRotatef


class TransformApplier:

    EXPLODE_DISTANCE: float = 8.0
    # When explode=1.0 objects are pushed 8.0 units from origin.
    # Scale is linear: explode=0.5 → 4.0 units pushed.

    def apply_global(self, rotation_y: float, scale: float) -> None:
        """Apply scene-wide scale then Y-rotation to the current MODELVIEW matrix.

        Call once per frame after camera setup, before drawing any objects.
        Order: scale first, then rotate — scale is applied in object space,
        rotation spins the entire scaled scene around the Y axis.

        Args:
            rotation_y: Y-axis rotation in degrees.
            scale:      Uniform scale factor (positive float).
        """
        glScalef(scale, scale, scale)
        glRotatef(rotation_y, 0.0, 1.0, 0.0)

    def get_explode_offset(
        self,
        world_position: np.ndarray,
        explode: float,
    ) -> np.ndarray:
        """Compute the per-object push offset for exploded-view rendering.

        Does NOT modify world_position. Returns the offset vector only;
        the caller is responsible for adding it temporarily during dispatch.

        Args:
            world_position: Current world position of the object, shape (3,).
            explode:        Explode factor in [0.0, 1.0].

        Returns:
            np.ndarray of shape (3,) — the offset to add to world_position.
        """
        if explode == 0.0:
            return np.zeros(3)

        norm = np.linalg.norm(world_position)
        if norm < 0.001:
            return np.zeros(3)  # sun (at origin) stays put

        direction = world_position / norm
        return direction * explode * self.EXPLODE_DISTANCE
