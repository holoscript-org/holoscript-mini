"""renderer/cylindrical/frame_builder.py
Assembles the (360, 18, 3) POV frame from a list of SceneObjects.
Imports: numpy, math, renderer.cylindrical.projector.
No OpenGL, no pyglet, no SceneState.
"""

from __future__ import annotations

import math

import numpy as np

from renderer.cylindrical.projector import (
    cartesian_to_angle_idx,
    y_to_led_idx,
    compute_scene_bounds,
    sample_sphere_surface,
    sample_cube_surface,
    get_n_samples,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _luminance(rgb: np.ndarray) -> float:
    """Perceptual luminance of an RGB float32 triplet."""
    return float(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])


# ---------------------------------------------------------------------------
# build_frame
# ---------------------------------------------------------------------------

def build_frame(objects: list, y_min: float, y_max: float) -> np.ndarray:
    """Convert a list of SceneObjects to a (360, 18, 3) uint8 POV frame.

    Uses a brightest-wins accumulation strategy so no pixel can be
    overwritten by a dimmer contribution.  Accumulates in float32,
    clips, and converts to uint8 at the very end.

    Args:
        objects: List of SceneObject instances with current world_position.
        y_min:   Lower vertical bound of the scene (from compute_scene_bounds).
        y_max:   Upper vertical bound of the scene.

    Returns:
        np.ndarray of shape (360, 18, 3), dtype uint8.
    """
    frame = np.zeros((360, 18, 3), dtype=np.float32)

    for obj in objects:
        if obj.type == "label":
            continue

        n_samples = get_n_samples(obj.size)
        wp = obj.world_position

        if obj.type == "sphere":
            points = sample_sphere_surface(
                float(wp[0]), float(wp[1]), float(wp[2]),
                obj.size, n_samples,
            )
        elif obj.type == "cube":
            points = sample_cube_surface(
                float(wp[0]), float(wp[1]), float(wp[2]),
                obj.size, n_samples,
            )
        else:
            # cylinder and ring: use sphere sampling as approximation
            points = sample_sphere_surface(
                float(wp[0]), float(wp[1]), float(wp[2]),
                obj.size, n_samples,
            )

        color_float = np.array(obj.color, dtype=np.float32) * 255.0

        for px, py, pz in points:
            angle_idx = cartesian_to_angle_idx(float(px), float(pz))
            led_pos   = y_to_led_idx(float(py), y_min, y_max)

            lower_led = int(math.floor(led_pos))
            upper_led = min(lower_led + 1, 17)
            frac      = led_pos - lower_led

            lower_contrib = color_float * (1.0 - frac)
            upper_contrib = color_float * frac

            # Brightest-wins for lower LED
            if _luminance(lower_contrib) > _luminance(frame[angle_idx, lower_led]):
                frame[angle_idx, lower_led] = lower_contrib

            # Brightest-wins for upper LED
            if _luminance(upper_contrib) > _luminance(frame[angle_idx, upper_led]):
                frame[angle_idx, upper_led] = upper_contrib

    return np.clip(frame, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# build_frame_from_scene
# ---------------------------------------------------------------------------

def build_frame_from_scene(objects: list) -> np.ndarray:
    """Convenience wrapper that auto-computes scene bounds before building.

    Args:
        objects: List of SceneObject instances.

    Returns:
        np.ndarray of shape (360, 18, 3), dtype uint8.
    """
    y_min, y_max = compute_scene_bounds(objects)
    return build_frame(objects, y_min, y_max)
