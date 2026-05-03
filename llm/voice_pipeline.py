"""llm/voice_pipeline.py

Phase 6 voice integration helpers.
Converts a text command into validated scene JSON via the staged pipeline:
planner -> parametric generator -> builder -> validator.
Also persists the resulting scene JSON to the renderer-consumed output paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.utils.logger import get_logger
from llm.planner import plan, ScenePlan
from llm.parametric_generator import generate_parametric_scene
from llm.builder import build_scene_json
from llm.validator import validate_member1

logger = get_logger("voice_pipeline")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT_PATHS = [
    _PROJECT_ROOT / "scene_grammar.json",
    _PROJECT_ROOT / "core" / "outputs" / "scene_grammar.json",
]


def generate_scene_from_command(command: str, intent: str = "NEW_SCENE") -> dict[str, Any]:
    """Generate validated scene JSON from a command using the staged pipeline."""
    plan_obj: ScenePlan | None = plan(command)
    if plan_obj is None:
        raise RuntimeError(f"planner failed for command: {command!r}")

    parametric_scene = generate_parametric_scene(plan_obj)
    scene_json = build_scene_json(plan_obj, parametric_scene, validate=True)
    try:
        validate_member1(scene_json)
    except Exception as e:
        # Save debug copy for inspection when validation fails (missing parents, etc.)
        debug_path = _PROJECT_ROOT / "core" / "outputs" / "failed_scene_debug.json"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_path, "w", encoding="utf-8") as fh:
            json.dump(scene_json, fh, indent=2)
        logger.error("voice_pipeline: validation failed - saved debug scene to %s", debug_path)
        # Re-raise the original exception so caller sees the error
        raise

    logger.info(
        "voice_pipeline: generated scene_type=%s num_objects=%d intent=%s",
        plan_obj.scene_type,
        len(scene_json.get("objects", [])),
        intent,
    )
    return scene_json


def persist_scene_outputs(scene_json: dict[str, Any]) -> list[Path]:
    """Write a scene JSON payload to the legacy and renderer-consumed output paths."""
    written_paths: list[Path] = []
    for path in _OUTPUT_PATHS:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(scene_json, handle, indent=2)
        written_paths.append(path)
        logger.info("voice_pipeline: wrote scene to %s", path)
    return written_paths


__all__ = ["generate_scene_from_command", "persist_scene_outputs"]
