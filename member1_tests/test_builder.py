#!/usr/bin/env python3
"""Phase 3 test harness for the Builder.

Runs planner -> parametric generator -> builder and validates the final
scene JSON against `llm.scene_schema.SceneSchema`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm.planner import plan
from llm.parametric_generator import generate_parametric_scene
from llm.builder import build_scene_json


TEST_COMMANDS = [
    "show a hydrogen atom",
    "create a solar system with 8 planets",
    "display a DNA double helix",
    "show a water molecule",
]


def main() -> int:
    print("=" * 70)
    print("Phase 3 Builder Test")
    print("=" * 70)

    failures = 0

    for command in TEST_COMMANDS:
        print(f"\nCommand: {command}")
        print("-" * 70)

        plan_obj = plan(command)
        if plan_obj is None:
            print("❌ FAILED: planner returned None")
            failures += 1
            continue

        parametric = generate_parametric_scene(plan_obj)

        try:
            scene_json = build_scene_json(plan_obj, parametric, validate=True)
        except Exception as e:
            print(f"❌ FAILED: builder validation failed: {e}")
            failures += 1
            continue

        objects = scene_json.get("objects", [])
        print(f"✓ Scene objects: {len(objects)}")

        if not objects:
            print("❌ FAILED: no objects produced by builder")
            failures += 1
            continue

        # Preview up to 4 objects
        preview = [
            {
                "id": o["id"],
                "type": o["type"],
                "position": [round(v, 3) for v in o["position"]],
                "animation": o["animation"],
            }
            for o in objects[:4]
        ]
        print(json.dumps(preview, indent=2))

    print("\n" + "=" * 70)
    if failures:
        print(f"Completed with {failures} failure(s)")
        return 1

    print("Completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
