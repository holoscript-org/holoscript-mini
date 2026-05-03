#!/usr/bin/env python3
"""Phase 6 test: voice command -> planner -> parametric -> builder -> validator -> persistence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm.voice_pipeline import generate_scene_from_command, persist_scene_outputs
from llm.validator import validate_member1


TEST_COMMANDS = [
    "show a hydrogen atom",
    "create a solar system with 8 planets",
    "display a DNA double helix",
]


def main() -> int:
    print("=" * 70)
    print("Phase 6 Voice Pipeline Test")
    print("=" * 70)

    failures = 0
    samples = []

    for command in TEST_COMMANDS:
        print(f"\nCommand: {command}")
        print("-" * 70)

        try:
            scene_json = generate_scene_from_command(command)
            validate_member1(scene_json)
            objects = scene_json.get("objects", [])
            print(f"✓ scene objects: {len(objects)}")
            print(f"✓ validated: {command}")

            samples.append({
                "command": command,
                "num_objects": len(objects),
                "scene": scene_json,
            })
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failures += 1

    output_paths = persist_scene_outputs(samples[-1]["scene"] if samples else {"objects": []}) if samples else []
    if output_paths:
        print(f"\n✓ persisted last sample to: {', '.join(str(path) for path in output_paths)}")

    output_file = PROJECT_ROOT / "integration_test_outputs" / "voice_pipeline_samples.json"
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(samples, handle, indent=2)

    print(f"✓ sample log written to: {output_file}")

    print("\n" + "=" * 70)
    if failures:
        print(f"Completed with {failures} failure(s)")
        return 1

    print("All Phase 6 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
