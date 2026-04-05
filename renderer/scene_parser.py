"""renderer/scene_parser.py
Standalone scene parser for the HoloScript Renderer module.
No imports from core/; no knowledge of SceneState.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_TYPES = {"sphere", "cube", "cylinder", "ring", "label"}


# ---------------------------------------------------------------------------
# SceneObject
# ---------------------------------------------------------------------------

@dataclass
class SceneObject:
    id: str
    type: str
    position: np.ndarray        # shape (3,), original from JSON
    color: tuple                # (R, G, B) floats 0.0-1.0
    emissive: bool              # True if id == "sun"
    is_orbiting: bool           # True if animation == "orbit"
    orbit_center: np.ndarray    # shape (3,)
    orbit_radius: float         # np.linalg.norm(position - orbit_center)
    orbit_speed: float
    size: float                 # assigned by default rules
    world_position: np.ndarray  # shape (3,), starts as position.copy()


# ---------------------------------------------------------------------------
# Size assignment
# ---------------------------------------------------------------------------

def _assign_size(obj_id: str, obj_type: str) -> float:
    if obj_id == "sun":
        return 2.0
    if obj_id in ("jupiter", "saturn"):
        return 1.0
    if obj_id in ("uranus", "neptune"):
        return 0.8
    if obj_type == "sphere":
        return 0.5
    if obj_type == "cube":
        return 0.5
    if obj_type == "cylinder":
        return 0.5
    if obj_type == "ring":
        return 1.0
    if obj_type == "label":
        return 0.0
    return 0.5


# ---------------------------------------------------------------------------
# parse_scene
# ---------------------------------------------------------------------------

def parse_scene(scene_dict: dict) -> List[SceneObject]:
    """Parse a raw scene dict (as produced by the LLM) into SceneObject instances.

    Args:
        scene_dict: Dict with an "objects" key, or None.

    Returns:
        List of valid SceneObject instances. Malformed entries are skipped
        with a printed warning; the function never raises.
    """
    if scene_dict is None or not isinstance(scene_dict, dict):
        print(f"[scene_parser] WARNING: scene_dict is {type(scene_dict).__name__!r}, expected dict. Returning [].")
        return []

    objects_raw = scene_dict.get("objects")
    if objects_raw is None or not isinstance(objects_raw, list):
        print("[scene_parser] WARNING: 'objects' key missing or not a list. Returning [].")
        return []

    result: List[SceneObject] = []

    for item in objects_raw:
        if not isinstance(item, dict):
            print(f"[scene_parser] WARNING: object entry is not a dict ({type(item).__name__!r}), skipping.")
            continue

        obj_id = item.get("id")
        obj_type = item.get("type")
        position_raw = item.get("position")
        color_raw = item.get("color")

        # Required fields check
        if obj_id is None or obj_type is None or position_raw is None or color_raw is None:
            label = obj_id if obj_id is not None else "<unknown>"
            print(f"[scene_parser] WARNING: object '{label}' missing required field(s) "
                  f"(id/type/position/color). Skipping.")
            continue

        # Type validation
        if obj_type not in VALID_TYPES:
            print(f"[scene_parser] WARNING: object '{obj_id}' has invalid type {obj_type!r} "
                  f"(must be one of {sorted(VALID_TYPES)}). Skipping.")
            continue

        # Optional fields with defaults
        animation = item.get("animation", "none")
        orbit_center_raw = item.get("orbit_center", [0.0, 0.0, 0.0])
        orbit_speed = float(item.get("orbit_speed", 0.0))

        # Array conversions
        position = np.array(position_raw, dtype=float)
        orbit_center = np.array(orbit_center_raw, dtype=float)

        orbit_radius = float(np.linalg.norm(position - orbit_center))
        is_orbiting = (animation == "orbit")
        emissive = (obj_id == "sun")
        size = _assign_size(obj_id, obj_type)
        world_position = position.copy()
        color = tuple(float(c) for c in color_raw)

        result.append(SceneObject(
            id=obj_id,
            type=obj_type,
            position=position,
            color=color,
            emissive=emissive,
            is_orbiting=is_orbiting,
            orbit_center=orbit_center,
            orbit_radius=orbit_radius,
            orbit_speed=orbit_speed,
            size=size,
            world_position=world_position,
        ))

    return result
