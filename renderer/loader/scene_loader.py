"""renderer/loader/scene_loader.py
Load a JSON scene file from disk and return parsed SceneObjects.

Priority order:
  1. core/outputs/scene_grammar.json  (Member 1 live output)
  2. core/assets/examples/<name>.json (named example)
  3. renderer/assets/solar_system.json (built-in default)
"""

from __future__ import annotations

import json
from pathlib import Path

from renderer.scene_parser import parse_scene, SceneObject

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CORE_OUTPUTS  = _PROJECT_ROOT / "core" / "outputs"
_CORE_EXAMPLES = _PROJECT_ROOT / "core" / "assets" / "examples"
_RENDERER_ASSETS = _PROJECT_ROOT / "renderer" / "assets"


def load_scene_from_file(filepath: str | Path) -> list[SceneObject]:
    """Load a JSON scene file and return the parsed list of SceneObjects."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[SceneLoader] WARNING: file not found: {filepath!r}")
        return []
    except json.JSONDecodeError as e:
        print(f"[SceneLoader] WARNING: invalid JSON in {filepath!r}: {e}")
        return []

    objects = parse_scene(data)
    print(f"[SceneLoader] Loaded {len(objects)} objects from {filepath}")
    return objects


def load_scene(scene_name: str | None = None) -> list[SceneObject]:
    """Load with priority: scene_grammar.json > named example > default scene."""
    grammar_path = _CORE_OUTPUTS / "scene_grammar.json"
    if grammar_path.exists():
        print("[SceneLoader] Loading from core/outputs/scene_grammar.json")
        return load_scene_from_file(grammar_path)

    if scene_name is not None:
        example_path = _CORE_EXAMPLES / f"{scene_name}.json"
        if example_path.exists():
            return load_scene_from_file(example_path)
        print(f"[SceneLoader] WARNING: example {scene_name!r} not found, falling back to default")

    return load_scene_from_file(_RENDERER_ASSETS / "solar_system.json")
