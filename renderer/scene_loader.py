"""renderer/scene_loader.py
Thin utility: load a JSON scene file from disk and return parsed SceneObjects.
No imports beyond stdlib + renderer.scene_parser.
"""

from __future__ import annotations

import json
import os

from renderer.scene_parser import parse_scene, SceneObject


def load_scene_from_file(filepath: str) -> list[SceneObject]:
    """Load a JSON scene file and return the parsed list of SceneObjects.

    Args:
        filepath: Path to a JSON file containing a scene dict.

    Returns:
        List of SceneObject instances. Returns [] on any error.
    """
    try:
        with open(filepath, "r") as f:
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
