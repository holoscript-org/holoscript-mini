"""llm/validator.py

Additional validation utilities for Member-1 strict rules. These checks are
kept separate from the pydantic `scene_schema` so we can progressively add
rules (hex colors, material checks, parent-cycle detection) without breaking
existing pipeline code.
"""

from __future__ import annotations

from typing import Dict, Any, Set
from llm import scene_schema


def _is_number_list_of_length(v, length: int) -> bool:
    if not isinstance(v, (list, tuple)):
        return False
    if len(v) != length:
        return False
    for x in v:
        if not isinstance(x, (int, float)):
            return False
    return True


def _detect_parent_cycle(objects: list[Dict[str, Any]]) -> bool:
    """Check if parent references form a cycle. Returns True if a cycle is found."""
    id_map = {obj.get("id"): obj for obj in objects}
    visited: Set[str] = set()
    rec_stack: Set[str] = set()

    def has_cycle(obj_id: str | None) -> bool:
        if obj_id is None:
            return False
        if obj_id in rec_stack:
            return True
        if obj_id in visited:
            return False

        visited.add(obj_id)
        rec_stack.add(obj_id)

        obj = id_map.get(obj_id)
        if obj:
            parent_id = obj.get("parent")
            if has_cycle(parent_id):
                return True

        rec_stack.discard(obj_id)
        return False

    for obj in objects:
        obj_id = obj.get("id")
        if obj_id and obj_id not in visited:
            if has_cycle(obj_id):
                return True

    return False


def validate_member1(scene: Dict[str, Any]) -> bool:
    """Run Member-1 style validation on a scene dict.

    Raises ValueError on validation failure.
    Returns True if scene is valid.
    """
    if not isinstance(scene, dict):
        raise ValueError("scene must be a dict")

    objects = scene.get("objects")
    if objects is None:
        raise ValueError("scene missing 'objects' key")
    if not isinstance(objects, list):
        raise ValueError("scene.objects must be a list")

    # reuse basic pydantic schema checks first
    try:
        scene_schema.SceneSchema.model_validate({"objects": objects})
    except Exception as e:
        raise ValueError(f"base schema validation failed: {e}")

    # Check for parent cycles
    if _detect_parent_cycle(objects):
        raise ValueError("parent references contain a cycle")

    seen_ids = set()
    seen_parents = set()

    for obj in objects:
        obj_id = obj.get("id")
        if not isinstance(obj_id, str) or not obj_id.strip():
            raise ValueError("each object must have a non-empty string 'id'")
        if obj_id in seen_ids:
            raise ValueError(f"duplicate object id: {obj_id}")
        seen_ids.add(obj_id)

        # position already checked by pydantic shape; ensure numeric ranges
        pos = obj.get("position")
        if not _is_number_list_of_length(pos, 3):
            raise ValueError(f"object {obj_id} has invalid position")

        color = obj.get("color")
        if not _is_number_list_of_length(color, 3):
            raise ValueError(f"object {obj_id} has invalid color format")
        for c in color:
            if not (0.0 <= float(c) <= 1.0):
                raise ValueError(f"object {obj_id} color components must be between 0.0 and 1.0")

        anim = obj.get("animation")
        if anim not in {"none", "orbit"}:
            raise ValueError(f"object {obj_id} has invalid animation value: {anim}")

        orbit_center = obj.get("orbit_center")
        if anim == "orbit":
            if not _is_number_list_of_length(orbit_center, 3):
                raise ValueError(f"object {obj_id} has animation 'orbit' but invalid orbit_center")

        orbit_speed = obj.get("orbit_speed")
        try:
            if float(orbit_speed) < 0.0:
                raise ValueError(f"object {obj_id} has negative orbit_speed")
        except Exception:
            raise ValueError(f"object {obj_id} has invalid orbit_speed")

        # Validate optional parent reference
        parent = obj.get("parent")
        if parent is not None:
            if not isinstance(parent, str) or not parent.strip():
                raise ValueError(f"object {obj_id} has invalid parent value")
            seen_parents.add(parent)

        # Validate optional scale
        scale = obj.get("scale")
        if scale is not None:
            if not _is_number_list_of_length(scale, 3):
                raise ValueError(f"object {obj_id} has invalid scale")
            for s in scale:
                if float(s) <= 0.0:
                    raise ValueError(f"object {obj_id} scale values must be positive")

        # Validate optional material
        material = obj.get("material")
        if material is not None:
            if not isinstance(material, str):
                raise ValueError(f"object {obj_id} material must be a string")

    # Check that all parents exist as object ids
    for parent_id in seen_parents:
        if parent_id not in seen_ids:
            raise ValueError(f"parent reference '{parent_id}' does not exist as an object id")

    return True


__all__ = ["validate_member1"]
