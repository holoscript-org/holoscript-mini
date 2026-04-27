"""renderer/animation.py
Orbital animation logic for the HoloScript Renderer.
Pure math — no pyglet, no OpenGL, no SceneState.
"""

from __future__ import annotations

from math import cos, sin

import numpy as np


class Animator:
    """Updates world_position of orbiting SceneObjects each frame.

    Keeps an internal simulation clock (sim_time) that advances by
    the real elapsed time (dt) every update unless frozen.
    """

    def __init__(self) -> None:
        self.sim_time: float = 0.0
        # SPEED_SCALE controls visual orbit speed.
        # 0.5 → orbit_speed=1.0 completes one lap every ~12.6 s of sim_time.
        self.SPEED_SCALE: float = 0.5

    def update(self, objects: list, dt: float, frozen: bool) -> None:
        """Recompute world_position for every orbiting object.

        Args:
            objects: List of SceneObject instances (modified in place).
            dt:      Elapsed time since last call, in seconds.
            frozen:  If True, do nothing and return immediately.
        """
        if frozen:
            return

        self.sim_time += dt

        for obj in objects:
            if not obj.is_orbiting:
                continue

            angle = obj.orbit_speed * self.sim_time * self.SPEED_SCALE
            world_x = obj.orbit_center[0] + obj.orbit_radius * cos(angle)
            world_z = obj.orbit_center[2] + obj.orbit_radius * sin(angle)
            world_y = obj.orbit_center[1]
            obj.world_position = np.array([world_x, world_y, world_z])

    def reset(self) -> None:
        """Reset simulation time to 0.0 (call when a new scene is loaded)."""
        self.sim_time = 0.0
