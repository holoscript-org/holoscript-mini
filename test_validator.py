#!/usr/bin/env python3
"""Quick validator test: run full pipeline and assert `validate_member1` passes."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm.planner import plan
from llm.parametric_generator import generate_parametric_scene
from llm.builder import build_scene_json
from llm.validator import validate_member1


TEST_COMMANDS = [
    "show a hydrogen atom",
    "create a solar system with 8 planets",
    "display a DNA double helix",
]


def main() -> int:
    failures = 0
    for command in TEST_COMMANDS:
        plan_obj = plan(command)
        if plan_obj is None:
            print(f"planner failed for: {command}")
            failures += 1
            continue
        param = generate_parametric_scene(plan_obj)
        scene = build_scene_json(plan_obj, param, validate=True)
        try:
            validate_member1(scene)
            print(f"✓ validate_member1: {command}")
        except Exception as e:
            print(f"✗ validate_member1 FAILED for {command}: {e}")
            failures += 1

    if failures:
        print(f"Completed with {failures} failure(s)")
        return 1
    print("All validations passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
