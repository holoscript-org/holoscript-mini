"""llm/builder.py

Phase 3 Builder: convert a ParametricScene + ScenePlan into the strict
SceneSchema expected by downstream systems (Member-1 style). This module
produces a JSON-friendly dict matching the Member1 contract with full
scene-level fields (name, camera, lights) and object-level fields
(geometry, material, animation, scale, label).

The builder assigns deterministic colors, geometries, and animations
based on the plan and placement hints.
"""

from __future__ import annotations

from typing import Any
import hashlib
import colorsys
from core.utils.logger import get_logger
from llm.parametric_generator import ParametricScene, PlacementSpec
from llm.planner import ScenePlan
from llm import scene_schema

logger = get_logger("builder")


def _rgb_to_hex(rgb: list[float]) -> str:
    """Convert RGB [0.0-1.0] to hex #rrggbb."""
    r = int(round(rgb[0] * 255)) & 0xFF
    g = int(round(rgb[1] * 255)) & 0xFF
    b = int(round(rgb[2] * 255)) & 0xFF
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_to_rgb(hex_color: str) -> list[float]:
    """Convert hex #rrggbb to RGB floats."""
    text = hex_color.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"invalid hex color: {hex_color!r}")
    return [
        round(int(text[0:2], 16) / 255.0, 4),
        round(int(text[2:4], 16) / 255.0, 4),
        round(int(text[4:6], 16) / 255.0, 4),
    ]


def _stable_int(*parts: str) -> int:
    data = "|".join(parts).encode("utf-8")
    return int(hashlib.sha1(data).hexdigest()[:8], 16)


def _role_to_type(role: str, metadata: dict[str, Any]) -> str:
    role_lower = str(role).lower()
    kind = str(metadata.get("kind", "")).lower()
    
    # Check both role name and metadata kind for type hints
    if "ring" in role_lower or "ring" in kind:
        return "ring"
    if kind in {"connector", "strand", "organic_segment"}:
        return "cylinder"
    if kind == "support":
        return "cube"
    if kind in {"label", "text"}:
        return "label"
    if "moon" in role_lower:
        return "sphere"  # Moons are spheres
    # default primitive
    return "sphere"


def _role_to_label(role: str) -> str | None:
    """Convert role to human-readable label or None."""
    label = role.replace("_", " ").strip().title()
    return label or None


def _resolve_palette(plan: ScenePlan) -> list[str]:
    if plan.color_palette:
        return list(plan.color_palette)

    seed = _stable_int(plan.scene_type, plan.layout_strategy, plan.camera_intent, plan.lighting_style, plan.description)
    palette: list[str] = []
    for index in range(4):
        hue = ((seed >> (index * 6)) & 0x3F) / 64.0
        saturation = 0.55 + (((seed >> (index * 5 + 3)) & 0x07) / 32.0)
        value = 0.72 + (((seed >> (index * 4 + 6)) & 0x03) / 10.0)
        red, green, blue = colorsys.hsv_to_rgb(hue, min(1.0, saturation), min(1.0, value))
        palette.append(_rgb_to_hex([red, green, blue]))
    return palette


def _palette_color(plan: ScenePlan, role: str, index: int) -> list[float]:
    palette = _resolve_palette(plan)
    if palette:
        chosen = palette[index % len(palette)]
        try:
            return _hex_to_rgb(chosen)
        except Exception:
            pass
    return _deterministic_color(role)


def _style_bonus(plan: ScenePlan) -> dict[str, float]:
    seed = _stable_int(plan.layout_strategy, plan.camera_intent, plan.lighting_style, ",".join(plan.style_hints))
    roughness = 0.2 + ((seed & 0xFF) / 255.0) * 0.6
    metalness = ((seed >> 8) & 0xFF) / 255.0 * 0.4
    return {"roughness": round(roughness, 3), "metalness": round(metalness, 3)}


def _deterministic_color(role: str) -> list[float]:
    """Map the role string to a deterministic RGB triple in [0.0, 1.0]."""
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


def _build_geometry(obj_type: str, role: str, size_scale: float = 1.0) -> dict[str, Any]:
    """Build a geometry object based on type."""
    if obj_type == "mesh":
        return {}
    if obj_type == "sphere":
        return {"type": "sphere", "radius": 0.5 * size_scale}
    elif obj_type == "cube":
        return {"type": "box", "width": 1.0 * size_scale, "height": 1.0 * size_scale, "depth": 1.0 * size_scale}
    elif obj_type == "cylinder":
        return {"type": "cylinder", "radius": 0.3 * size_scale, "height": 1.0 * size_scale}
    elif obj_type == "ring":
        return {"type": "ring", "innerRadius": 2.9 * size_scale, "outerRadius": 3.1 * size_scale, "thetaSegments": 64}
    elif obj_type == "label":
        return {"type": "plane", "width": 2.0 * size_scale, "height": 1.0 * size_scale}
    else:
        return {"type": "sphere", "radius": 0.5 * size_scale}


def _build_material(color_rgb: list[float], obj_type: str = "sphere", plan: ScenePlan | None = None) -> dict[str, Any]:
    """Build a material object with proper color, roughness, metalness, emissive."""
    color_hex = _rgb_to_hex(color_rgb)
    bonus = _style_bonus(plan) if plan is not None else {"roughness": 0.4, "metalness": 0.08}
    
    # Assign material properties based on object type
    if obj_type == "ring":
        return {
            "type": "standard",
            "color": color_hex,
            "roughness": bonus["roughness"],
            "metalness": 0.0,
            "opacity": 0.6,
            "transparent": True,
        }
    elif obj_type in {"strand", "cylinder"}:
        return {
            "type": "standard",
            "color": color_hex,
            "roughness": bonus["roughness"],
            "metalness": bonus["metalness"],
            "opacity": 1.0,
            "transparent": False,
        }
    else:
        # Default material for spheres and other objects
        return {
            "type": "standard",
            "color": color_hex,
            "roughness": bonus["roughness"],
            "metalness": bonus["metalness"],
            "opacity": 1.0,
            "transparent": False,
        }


def _build_animation(animation_type: str, orbit_center: list[float], orbit_speed: float) -> dict[str, Any]:
    """Build a structured animation object."""
    if animation_type == "orbit":
        return {
            "type": "orbit",
            "center": orbit_center,
            "speed": orbit_speed,
        }
    elif animation_type == "spin":
        return {
            "type": "spin",
            "speed": 1.0,
        }
    else:
        return {"type": "none"}


def _build_scene_name(plan: ScenePlan) -> str:
    """Generate a human-readable scene name from the plan."""
    if plan.description.strip():
        return plan.description.strip()[:1].upper() + plan.description.strip()[1:]
    return plan.scene_type.replace("_", " ").title()


def _build_camera(plan: ScenePlan, num_objects: int) -> dict[str, Any]:
    """Build a camera config from plan hints and composition size."""
    distance = max(10.0, 5.0 + num_objects * 0.6)
    intent = plan.camera_intent
    seed = _stable_int(plan.scene_type, plan.layout_strategy, plan.description, plan.camera_intent)
    offset_x = (((seed >> 0) & 0x0F) - 8) / 8.0
    offset_y = (((seed >> 4) & 0x0F) - 8) / 10.0
    offset_z = (((seed >> 8) & 0x0F) - 8) / 8.0

    if intent == "close":
        return {"position": [offset_x, max(2.5, distance * 0.25) + offset_y, max(6.0, distance * 0.4) + offset_z], "target": [0, 0, 0], "fov": 70}
    if intent == "wide":
        return {"position": [offset_x, distance * 0.45 + offset_y, distance * 1.35 + offset_z], "target": [0, 0, 0], "fov": 55}
    if intent == "top_down":
        return {"position": [offset_x, max(18.0, distance * 1.2), offset_z * 0.1], "target": [0, 0, 0], "fov": 60}
    if intent == "cinematic":
        return {"position": [distance * 0.45 + offset_x, distance * 0.18 + offset_y, distance + offset_z], "target": [0, 0, 0], "fov": 52}

    layout_factor = 0.25 + ((seed >> 12) & 0x0F) / 40.0
    return {"position": [distance * layout_factor + offset_x, distance * 0.55 + offset_y, distance + offset_z], "target": [0, 0, 0], "fov": 60}


def _build_lights(plan: ScenePlan) -> list[dict[str, Any]]:
    """Build a lights array from lighting style and style hints."""
    mood = plan.lighting_style
    seed = _stable_int(plan.scene_type, plan.layout_strategy, plan.camera_intent, plan.lighting_style, ",".join(plan.style_hints))
    warm_bias = mood == "warm" or "warm" in plan.style_hints
    cool_bias = mood == "cool" or "cool" in plan.style_hints
    dramatic_bias = mood == "dramatic" or "dramatic" in plan.style_hints
    neon_bias = mood == "neon" or any(hint in {"glowing", "bioluminescent", "neon"} for hint in plan.style_hints)
    clinical_bias = mood == "clinical"

    ambient = 0.18 + ((seed & 0x0F) / 100.0)
    main_intensity = 0.8 + (((seed >> 4) & 0x0F) / 20.0)
    fill_intensity = 0.7 + (((seed >> 8) & 0x0F) / 30.0)

    if warm_bias:
        key_color = "#ffd39a"
        fill_color = "#ff9966"
    elif cool_bias:
        key_color = "#dce9ff"
        fill_color = "#88ccff"
    elif neon_bias:
        key_color = "#88ccff"
        fill_color = "#ff99aa"
    elif clinical_bias:
        key_color = "#ffffff"
        fill_color = "#ddeeff"
    elif dramatic_bias:
        key_color = "#ffffff"
        fill_color = "#ffb36b"
    else:
        key_color = "#ffffff"
        fill_color = "#ffd8b0"

    lights = [
        {"type": "ambient", "intensity": round(ambient, 3), "color": "#ffffff"},
        {"type": "directional", "intensity": round(main_intensity, 3), "color": key_color, "position": [8, 10, 8], "castShadow": True},
        {"type": "point", "intensity": round(fill_intensity, 3), "color": fill_color, "position": [0, 2, 0]},
    ]

    if neon_bias or dramatic_bias:
        lights.append({"type": "point", "intensity": round(0.6 + ((seed >> 12) & 0x0F) / 20.0, 3), "color": fill_color, "position": [0, 1, -10]})

    return lights


def build_scene_json(plan: ScenePlan, parametric: ParametricScene, validate: bool = True) -> dict:
    """Convert `ParametricScene` into a Member1-compliant scene dict.

    Args:
        plan: The abstract scene plan.
        parametric: Deterministic placements from `generate_parametric_scene`.
        validate: If True, validate the result against `llm.scene_schema.SceneSchema`.

    Returns:
        A JSON-serializable dict matching the Member1 contract with scene-level
        and object-level fields.
    """

    objects: list[dict[str, Any]] = []

    for placement in parametric.placements:
        obj_id = placement.role
        obj_type = "mesh" if plan.use_mesh and "mesh" in placement.role.lower() else _role_to_type(placement.role, placement.metadata)
        pos = [float(placement.position[0]), float(placement.position[1]), float(placement.position[2])]
        
        # Try component-specific color first, then fall back to palette
        color_rgb: list[float] = [0.5, 0.5, 0.5]  # default gray
        if plan.component_colors and placement.role in plan.component_colors:
            # Component has a specific color assigned
            hex_color = plan.component_colors[placement.role]
            try:
                color_rgb = _hex_to_rgb(hex_color)
            except Exception:
                color_rgb = _palette_color(plan, placement.role, placement.index)
        else:
            # Fall back to palette
            color_rgb = _palette_color(plan, placement.role, placement.index)

        # Get component-specific size, default to 1.0
        size_scale = 1.0
        if plan.component_sizes and placement.role in plan.component_sizes:
            size_scale = plan.component_sizes[placement.role]

        # Animation decision: orbit if plan allows it and the placement suggests an orbit.
        wants_orbit = bool(placement.orbit_center) or placement.metadata.get("kind") in {"orbiting", "moon", "planet"}
        animation_type = "orbit" if ("orbit" in plan.animation_types and wants_orbit) else "none"

        orbit_center = [0.0, 0.0, 0.0]
        if placement.orbit_center:
            orbit_center = [float(placement.orbit_center[0]), float(placement.orbit_center[1]), float(placement.orbit_center[2])]

        orbit_speed = _deterministic_orbit_speed(placement.role, placement.index) if animation_type == "orbit" else 0.0

        # Build the full object with all Member1 fields
        obj = {
            "id": str(obj_id),
            "type": "mesh" if obj_type == "mesh" else "primitive",
            "position": pos,
            "scale": [size_scale, size_scale, size_scale],
            "material": _build_material(color_rgb, obj_type, plan),
            "animation": _build_animation(animation_type, orbit_center, orbit_speed),
        }

        if obj_type == "mesh":
            obj["model"] = str(placement.metadata.get("model", f"/models/{plan.scene_type}.glb"))
        else:
            obj["geometry"] = _build_geometry(obj_type, placement.role, size_scale)
        
        # Add label if available
        label = _role_to_label(placement.role)
        if label:
            obj["label"] = label
        
        # Add parent if specified in component_parent map from plan
        if plan.component_parent and placement.role in plan.component_parent:
            obj["parent"] = plan.component_parent[placement.role]
        elif "parent" in placement.metadata:
            # Also respect parent from metadata (fallback)
            obj["parent"] = placement.metadata["parent"]

        objects.append(obj)

    # Post-process parent references: planner may refer to semantic parent names
    # that don't exactly match generated object ids (e.g., planner says "strand_2"
    # while parametric produced "strand_left_2" / "strand_right_2"). Attempt
    # to resolve such parent names to the best matching object id using
    # exact, prefix, or substring heuristics to avoid validation failures.
    ids = {o["id"] for o in objects}
    for obj in objects:
        parent = obj.get("parent")
        if parent is None:
            continue
        if parent in ids:
            continue
        # Try to find an object id that starts with the parent name
        candidates = [i for i in ids if i == parent or i.startswith(parent + "_")]
        if not candidates:
            # Fallback: any id that contains the parent token
            candidates = [i for i in ids if parent in i]
        if candidates:
            # Choose the shortest candidate (prefer exact-like) to reduce ambiguity
            chosen = sorted(candidates, key=lambda s: (len(s), s))[0]
            obj["parent"] = chosen
            logger.info("builder: remapped parent '%s' -> '%s' for object '%s'", parent, chosen, obj["id"])
        else:
            logger.warning("builder: unresolved parent reference '%s' for object '%s'", parent, obj.get("id"))

    # Build scene-level fields
    scene_dict = {
        "name": _build_scene_name(plan),
        "camera": _build_camera(plan, len(objects)),
        "lights": _build_lights(plan),
        "objects": objects,
    }

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
