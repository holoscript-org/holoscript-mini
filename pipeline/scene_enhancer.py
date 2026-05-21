"""
Groq-powered scene enhancement pass.

Enhances a deterministic base scene without removing grounded objects.
All output must conform to gui/lib/sceneFactory.ts via scene_validator.
"""
from __future__ import annotations

import json
import os
from typing import Any

from llm.groq_client import generate_raw
from pipeline.scene_validator import DEFAULT_CAMERA, DEFAULT_LIGHTS, validate_scene

SYSTEM_PROMPT = """
You are a scene enhancer. Output ONLY a JSON object that conforms to the scene schema.

Hard rules:
- Preserve ALL existing objects from the base scene (do not delete them).
- Do NOT change object ids or remove objects.
- Resolved entities are atomic. Do NOT split or reinterpret semantic meaning.
- Do NOT introduce new semantic entities or new mesh assets.
- Do NOT invent model paths. Only use provided assets.
- Preserve model paths, geometry, and object transforms (position/rotation/scale/parent).
- You MAY refine materials, animations, lights, and camera.
- When changing appearance, use materialOverrides instead of replacing models.
- Output valid JSON only. No markdown. No explanations.
- Omit any missing/unknown fields; never emit null values.

Schema summary (sceneFactory.ts):
- scene: { name?, objects[], lights[], camera }
- object: { id, type: primitive|mesh, position, material, animation?, geometry? (primitive), model? (mesh), parent?, rotation?, scale?, label?, materialOverrides? }
- material: { type:"standard", color:"#rrggbb", roughness 0-1, metalness 0-1, opacity?, transparent?, map?, normalMap?, roughnessMap?, metalnessMap?, emissive?, emissiveMap?, emissiveIntensity? }
- animation: { type:"none"|"orbit"|"spin", center?, center_ref?, axis?, speed?, phase? }
- lights: ambient|directional|point|spot; { type, intensity, color?, position?, castShadow? }
- camera: { position, target, fov? }
""".strip()


def enhance_scene(
    transcript: str,
    intent: dict[str, Any],
    components: dict[str, Any],
    base_scene: dict[str, Any],
    style_hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return _strip_none(base_scene)

    prompt = _build_prompt(transcript, intent, components, base_scene, style_hints)
    raw = generate_raw(prompt, SYSTEM_PROMPT)
    if not raw:
        return _strip_none(base_scene)

    candidate = _parse_json_object(raw)
    if not isinstance(candidate, dict):
        return _strip_none(base_scene)

    merged = _merge_scene(base_scene, candidate)
    vr = validate_scene(merged)
    final = vr.get("scene") if vr.get("scene") else merged
    return _strip_none(final)


def _build_prompt(
    transcript: str,
    intent: dict[str, Any],
    components: dict[str, Any],
    base_scene: dict[str, Any],
    style_hints: dict[str, Any] | None,
) -> str:
    assets = []
    for item in components.get("assets", []):
        assets.append(
            {
                "concept": item.get("concept"),
                "asset_id": item.get("asset_id"),
                "asset_src": item.get("asset_src"),
                "category": item.get("category"),
                **({"tags": item.get("tags")} if item.get("tags") else {}),
            }
        )

    available_assets = [
        {
            "concept": item.get("concept"),
            "asset_id": item.get("asset_id"),
            "asset_src": item.get("asset_src"),
        }
        for item in components.get("assets", [])
        if item.get("asset_src")
    ]

    payload = {
        "transcript": transcript,
        "semantic_intent": intent,
        "retrieved_assets": assets,
        "available_assets": available_assets,
        "style_hints": style_hints or _extract_style_hints(transcript),
        "base_scene": base_scene,
    }
    return f"{SYSTEM_PROMPT}\n\nINPUT:\n{json.dumps(payload, indent=2)}\n\nOUTPUT:\n"


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    try:
        text = raw.strip()
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1][4:] if parts[1].startswith("json") else parts[1]
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end <= start:
            return None
        return json.loads(text[start:end])
    except Exception:
        return None


def _merge_scene(base_scene: dict[str, Any], enhanced: dict[str, Any]) -> dict[str, Any]:
    base_objects = {obj.get("id"): obj for obj in base_scene.get("objects", []) if isinstance(obj, dict)}
    merged_objects: list[dict[str, Any]] = []
    seen: set[str] = set()

    for obj in enhanced.get("objects", []) if isinstance(enhanced.get("objects"), list) else []:
        if not isinstance(obj, dict):
            continue
        obj_id = obj.get("id")
        if not isinstance(obj_id, str) or not obj_id.strip():
            continue
        if obj_id in seen:
            continue
        base_obj = base_objects.get(obj_id)
        if base_obj:
            # Merge enhanced fields onto the base object
            merged = dict(base_obj)
            overrides = obj.get("materialOverrides") if isinstance(obj.get("materialOverrides"), dict) else None
            if overrides:
                merged["material"] = _apply_material_overrides(merged.get("material"), overrides)
            for key in ["material", "animation", "label", "position", "scale", "rotation", "geometry", "parent"]:
                val = obj.get(key)
                if val is not None:
                    merged[key] = val
            if _is_valid_object(merged):
                merged_objects.append(merged)
            else:
                merged_objects.append(base_obj)
        else:
            # Groq introduced a new object — accept it if valid
            if _is_valid_object(obj):
                merged_objects.append(obj)
        seen.add(obj_id)

    # Keep any base objects Groq did not mention
    for obj_id, obj in base_objects.items():
        if obj_id not in seen:
            merged_objects.append(obj)

    scene: dict[str, Any] = {
        "objects": merged_objects,
        "lights": enhanced.get("lights") if isinstance(enhanced.get("lights"), list) else base_scene.get("lights"),
        "camera": enhanced.get("camera") if isinstance(enhanced.get("camera"), dict) else base_scene.get("camera"),
    }
    if isinstance(enhanced.get("name"), str):
        scene["name"] = enhanced.get("name")
    elif isinstance(base_scene.get("name"), str):
        scene["name"] = base_scene.get("name")
    return scene


def _is_valid_object(obj: dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    probe = {
        "objects": [obj],
        "lights": DEFAULT_LIGHTS,
        "camera": DEFAULT_CAMERA,
    }
    vr = validate_scene(probe)
    return not vr.get("fatal") and not vr.get("errors") and bool(vr.get("scene", {}).get("objects"))


def _apply_material_overrides(base_material: Any, overrides: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(base_material, dict):
        base_material = {"type": "standard", "color": "#888888", "roughness": 0.6, "metalness": 0.1}
    merged = dict(base_material)
    for key in [
        "color",
        "roughness",
        "metalness",
        "opacity",
        "transparent",
        "map",
        "normalMap",
        "roughnessMap",
        "metalnessMap",
        "emissive",
        "emissiveMap",
        "emissiveIntensity",
    ]:
        if key in overrides:
            merged[key] = overrides[key]
    return merged


def _extract_style_hints(transcript: str) -> dict[str, Any]:
    text = (transcript or "").lower()
    color_map = {
        "red": "#b91c1c",
        "crimson": "#dc2626",
        "pink": "#ec4899",
        "blue": "#2563eb",
        "green": "#16a34a",
        "gold": "#d97706",
        "yellow": "#f59e0b",
        "white": "#f8fafc",
        "black": "#0f172a",
        "silver": "#94a3b8",
    }
    requested = [color_map[word] for word in color_map if word in text]
    hints: dict[str, Any] = {}
    if requested:
        hints["requested_colors"] = requested
    if "heart" in text:
        hints.setdefault("suggested_overrides", []).append(
            {"match": "heart", "materialOverrides": {"color": "#b91c1c"}}
        )
    return hints


def _strip_none(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {k: _strip_none(v) for k, v in value.items() if v is not None}
        return {k: v for k, v in cleaned.items() if v is not None}
    if isinstance(value, list):
        return [_strip_none(v) for v in value if v is not None]
    return value
