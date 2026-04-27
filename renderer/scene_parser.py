"""renderer/scene_parser.py
Standalone scene parser for the HoloScript Renderer module.
No imports from core/; no knowledge of SceneState.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_TYPES = {"sphere", "cube", "cylinder", "ring", "label", "mesh"}


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
    surface_style: str = "plain"
    secondary_color: tuple = (1.0, 1.0, 1.0)
    bands: list = None
    ring: dict = None
    size_override: float = None
    mesh_file: Optional[str] = None  # path to .obj file; only used when type == "mesh"
    scale: float = 1.0               # uniform scale applied to mesh geometry
    label: Optional[str] = None      # display name; drawn as billboard text if set


# ---------------------------------------------------------------------------
# Size assignment
# ---------------------------------------------------------------------------

def _assign_size(obj_type: str) -> float:
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
        surface_style = str(item.get("surface_style", "plain"))

        secondary_raw = item.get("secondary_color", [1.0, 1.0, 1.0])
        try:
            secondary_color = tuple(float(c) for c in secondary_raw)
            if len(secondary_color) != 3:
                raise ValueError("secondary_color must have 3 elements")
        except Exception:
            print(f"[scene_parser] WARNING: object '{obj_id}' has invalid secondary_color; using default.")
            secondary_color = (1.0, 1.0, 1.0)

        bands = item.get("bands")
        if bands is not None and not isinstance(bands, list):
            print(f"[scene_parser] WARNING: object '{obj_id}' has non-list bands; ignoring.")
            bands = None

        ring = item.get("ring")
        if ring is not None and not isinstance(ring, dict):
            print(f"[scene_parser] WARNING: object '{obj_id}' has non-dict ring; ignoring.")
            ring = None

        # Array conversions
        position = np.array(position_raw, dtype=float)
        orbit_center = np.array(orbit_center_raw, dtype=float)

        orbit_radius = float(np.linalg.norm(position - orbit_center))
        is_orbiting = (animation == "orbit")
        size_override = None
        if "size" in item:
            try:
                size_override = float(item.get("size"))
            except (TypeError, ValueError):
                print(f"[scene_parser] WARNING: object '{obj_id}' has invalid size override; using defaults.")
                size_override = None

        size = size_override if size_override is not None else _assign_size(obj_type)
        emissive = bool(item.get("emissive", False))

        # Mesh-specific fields
        mesh_file: Optional[str] = None
        scale: float = 1.0
        if obj_type == "mesh":
            mesh_file = item.get("mesh_file")
            if not mesh_file:
                print(f"[scene_parser] WARNING: mesh object '{obj_id}' missing 'mesh_file'. Skipping.")
                continue
            try:
                scale = float(item.get("scale", 1.0))
            except (TypeError, ValueError):
                print(f"[scene_parser] WARNING: object '{obj_id}' has invalid scale; using 1.0.")
                scale = 1.0

        label: Optional[str] = item.get("label")

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
            surface_style=surface_style,
            secondary_color=secondary_color,
            bands=bands,
            ring=ring,
            size_override=size_override,
            mesh_file=mesh_file,
            scale=scale,
            label=label,
        ))

    return result
