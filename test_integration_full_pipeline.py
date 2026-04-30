#!/usr/bin/env python3
"""Comprehensive integration test: Full pipeline with output samples."""

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
from llm.validator import validate_member1


TEST_COMMANDS = [
    "show a hydrogen atom",
    "create a solar system with 8 planets",
    "display a DNA double helix",
    "show a water molecule",
    "make a crystalline lattice",
]


def main() -> int:
    print("=" * 70)
    print("Full Integration Test: Planner → Parametric → Builder → Validator")
    print("=" * 70)

    output_dir = PROJECT_ROOT / "integration_test_outputs"
    output_dir.mkdir(exist_ok=True)

    failures = 0
    samples = []

    for command in TEST_COMMANDS:
        print(f"\nProcessing: {command}")
        print("-" * 70)

        # Phase 1: Planner
        plan_obj = plan(command)
        if plan_obj is None:
            print(f"❌ Planner failed")
            failures += 1
            continue

        print(f"✓ Planner: scene_type={plan_obj.scene_type}, num_objects={plan_obj.num_objects}")

        # Phase 2: Parametric Generator
        parametric = generate_parametric_scene(plan_obj)
        print(f"✓ Parametric: {len(parametric.placements)} placements")

        # Phase 3: Builder
        scene_json = build_scene_json(plan_obj, parametric, validate=True)
        objects = scene_json.get("objects", [])
        print(f"✓ Builder: {len(objects)} objects")

        # Phase 4: Validator
        try:
            validate_member1(scene_json)
            print(f"✓ Validator: Member-1 strict validation passed")
        except Exception as e:
            print(f"❌ Validator: {e}")
            failures += 1
            continue

        # Save sample output
        sample = {
            "command": command,
            "scene_type": plan_obj.scene_type,
            "num_objects": len(objects),
            "scene": scene_json,
        }
        samples.append(sample)

    # Save all samples to a JSON file
    output_file = output_dir / "integration_samples.json"
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Integration test outputs saved to: {output_file}")
    print(f"Total samples generated: {len(samples)}")

    if failures:
        print(f"Completed with {failures} failure(s)")
        return 1

    print("All integration tests passed ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
