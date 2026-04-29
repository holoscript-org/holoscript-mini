"""llm/builder.py

Phase 3 Builder: convert a ParametricScene + ScenePlan into the strict
SceneSchema expected by downstream systems (Member-1 style). This module
produces a simple JSON-friendly dict matching `llm.scene_schema.SceneSchema`.

The builder is conservative: it maps placement metadata to a small set of
primitive types, assigns deterministic colors, and sets animation/orbit
parameters based on the plan and placement hints.
"""

from __future__ import annotations

from typing import Any
from core.utils.logger import get_logger
from llm.parametric_generator import ParametricScene, PlacementSpec
from llm.planner import ScenePlan
from llm import scene_schema

logger = get_logger("builder")


def _role_to_type(role: str, metadata: dict[str, Any]) -> str:
    kind = str(metadata.get("kind", "")).lower()
    if "ring" in kind:
        return "ring"
    if kind in {"connector", "strand", "organic_segment"}:
        return "cylinder"
    if kind == "support":
        return "cube"
    if kind in {"label", "text"}:
        return "label"
    # default primitive
    return "sphere"


def _deterministic_color(role: str) -> list[float]:
    # Map the role string to a deterministic RGB triple in [0.0, 1.0].
    h = abs(hash(role))
    r = ((h >> 0) & 0xFF) / 255.0
    g = ((h >> 8) & 0xFF) / 255.0
    b = ((h >> 16) & 0xFF) / 255.0
    # Ensure no component is exactly 0.0 to keep objects visible.
    return [max(0.03, round(float(r), 4)), max(0.03, round(float(g), 4)), max(0.03, round(float(b), 4))]


def _deterministic_orbit_speed(role: str, index: int) -> float:
    h = abs(hash(role))
    # produce a small orbit speed in a predictable range
    base = 0.05
    extra = (h % 50) / 1000.0  # 0.0 - 0.049
    step = (index % 5) * 0.005
    return round(base + extra + step, 4)


def build_scene_json(plan: ScenePlan, parametric: ParametricScene, validate: bool = True) -> dict:
    """Convert `ParametricScene` into a dict matching `SceneSchema`.

    Args:
        plan: The abstract scene plan.
        parametric: Deterministic placements from `generate_parametric_scene`.
        validate: If True, validate the result against `llm.scene_schema.SceneSchema`.

    Returns:
        A JSON-serializable dict with key `objects` containing scene objects.
    """

    objects: list[dict[str, Any]] = []

    for placement in parametric.placements:
        obj_id = placement.role
        obj_type = _role_to_type(placement.role, placement.metadata)
        pos = [float(placement.position[0]), float(placement.position[1]), float(placement.position[2])]
        color = _deterministic_color(placement.role)

        # Animation decision: orbit if plan allows it and the placement suggests an orbit.
        wants_orbit = bool(placement.orbit_center) or placement.metadata.get("kind") in {"orbiting", "moon", "planet"}
        animation = "orbit" if ("orbit" in plan.animation_types and wants_orbit) else "none"

        orbit_center = [0.0, 0.0, 0.0]
        if placement.orbit_center:
            orbit_center = [float(placement.orbit_center[0]), float(placement.orbit_center[1]), float(placement.orbit_center[2])]

        orbit_speed = _deterministic_orbit_speed(placement.role, placement.index) if animation == "orbit" else 0.0

        obj = {
            "id": str(obj_id),
            "type": obj_type,
            "position": pos,
            "color": color,
            "animation": animation,
            "orbit_center": orbit_center,
            "orbit_speed": float(orbit_speed),
        }

        objects.append(obj)

    scene_dict = {"objects": objects}

    if validate:
        try:
            # Validate with pydantic model; will raise on invalid shapes/values.
            validated = scene_schema.SceneSchema.model_validate(scene_dict)
            logger.info("builder: validated scene with %d objects", len(objects))
            # Return the dict form of the validated model to keep types normalized.
            return validated.model_dump()
        except Exception as e:
            logger.error("builder: scene validation failed: %s", e)
            raise

    return scene_dict


__all__ = ["build_scene_json"]
