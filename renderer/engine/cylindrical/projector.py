"""renderer/cylindrical/projector.py
Pure coordinate-conversion functions for the cylindrical POV projection.
Imports: numpy and math only. No OpenGL, no pyglet, no SceneState.
"""

from __future__ import annotations

import math

import numpy as np


def cartesian_to_angle_idx(x: float, z: float) -> int:
    """Convert (x, z) world position to angular index 0-359.

    theta=0  → positive X axis → angle_idx=0
    theta=90 → positive Z axis → angle_idx=90

    Uses atan2(z, x) so that the standard mathematical convention maps
    cleanly onto the 360-slot angle array.
    """
    theta = math.atan2(z, x)
    if theta < 0:
        theta += 2 * math.pi
    angle_idx = int(math.floor(theta / (2 * math.pi) * 360))
    return max(0, min(359, angle_idx))


def y_to_led_idx(y: float, y_min: float, y_max: float) -> float:
    """Map Y coordinate to fractional LED index 0.0-17.0.

    Returns a float so callers can interpolate between adjacent LEDs.
    If y_max == y_min (flat scene): returns 8.5 (vertical center).
    """
    if y_max == y_min:
        return 8.5
    led_pos = (y - y_min) / (y_max - y_min) * 18.0
    return max(0.0, min(17.0, led_pos))


def compute_scene_bounds(objects: list) -> tuple:
    """Find the vertical (Y-axis) extent of the entire scene.

    Returns (y_min, y_max) considering each object's world_position ± size.
    Guarantees y_max > y_min so y_to_led_idx never divides by zero.
    """
    if not objects:
        return (-1.0, 1.0)

    all_lowers = [obj.world_position[1] - obj.size for obj in objects]
    all_uppers = [obj.world_position[1] + obj.size for obj in objects]
    y_min = float(min(all_lowers))
    y_max = float(max(all_uppers))

    if y_min == y_max:
        return (y_min - 1.0, y_max + 1.0)
    return (y_min, y_max)


def sample_sphere_surface(
    cx: float, cy: float, cz: float,
    radius: float,
    n_samples: int,
) -> np.ndarray:
    """Sample n_samples points approximately uniformly on a sphere surface.

    Uses a phi × theta grid in spherical coordinates.
    Returns shape (N, 3), dtype float64.
    """
    k = int(math.floor(math.sqrt(n_samples)))
    k = max(k, 2)

    phis = np.linspace(0, math.pi, k)
    thetas = np.linspace(0, 2 * math.pi, k)

    points = []
    for phi in phis:
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        for theta in thetas:
            x = cx + radius * sin_phi * math.cos(theta)
            y = cy + radius * cos_phi
            z = cz + radius * sin_phi * math.sin(theta)
            points.append((x, y, z))

    return np.array(points, dtype=np.float64)


def sample_cube_surface(
    cx: float, cy: float, cz: float,
    half_edge: float,
    n_samples: int,
) -> np.ndarray:
    """Sample points on all 6 faces of an axis-aligned cube.

    Distributes n_samples // 6 points per face on a 2-D grid.
    Returns shape (N, 3), dtype float64.
    """
    n_per_face = max(1, n_samples // 6)
    k = max(2, int(math.floor(math.sqrt(n_per_face))))
    ts = np.linspace(-half_edge, half_edge, k)

    faces = [
        # (face_center_offset_axis, axis0, axis1)
        # +X
        (lambda a, b: (cx + half_edge, cy + a, cz + b)),
        # -X
        (lambda a, b: (cx - half_edge, cy + a, cz + b)),
        # +Y
        (lambda a, b: (cx + a, cy + half_edge, cz + b)),
        # -Y
        (lambda a, b: (cx + a, cy - half_edge, cz + b)),
        # +Z
        (lambda a, b: (cx + a, cy + b, cz + half_edge)),
        # -Z
        (lambda a, b: (cx + a, cy + b, cz - half_edge)),
    ]

    points = []
    for face_fn in faces:
        for a in ts:
            for b in ts:
                points.append(face_fn(a, b))

    return np.array(points, dtype=np.float64)


def get_n_samples(size: float) -> int:
    """Compute number of surface samples for an object of the given size.

    Scales linearly with size; clamped to [50, 800].
    """
    n = int(size * 80)
    return max(50, min(n, 800))
