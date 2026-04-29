#!/usr/bin/env python3
"""Phase 2 test harness for the parametric generator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm.planner import plan
from llm.parametric_generator import generate_parametric_scene


TEST_COMMANDS = [
    "show a hydrogen atom",
    "create a solar system with 8 planets",
    "display a DNA double helix",
    "show a water molecule",
    "make a crystalline lattice",
]


def main() -> int:
    print("=" * 70)
    print("Phase 2 Parametric Generator Test")
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
        placements = parametric.placements

        print(f"✓ Scene Type: {parametric.scene_type}")
        print(f"✓ Placements: {len(placements)}")
        print(f"✓ Notes: {', '.join(parametric.notes) if parametric.notes else '(none)'}")

        if not placements:
            print("❌ FAILED: no placements generated")
            failures += 1
            continue

        if len(placements) != plan_obj.num_objects:
            print(f"⚠ Count mismatch: plan asked for {plan_obj.num_objects}, generator produced {len(placements)}")
        else:
            print("✓ Placement count matches plan")

        preview = [
            {
                "role": placement.role,
                "index": placement.index,
                "position": [round(value, 3) for value in placement.position],
                "parent": placement.parent,
                "metadata": placement.metadata,
            }
            for placement in placements[: min(5, len(placements))]
        ]
        print(json.dumps(preview, indent=2))

        # Basic sanity checks.
        for placement in placements:
            if len(placement.position) != 3:
                print(f"❌ FAILED: bad position length for {placement.role}")
                failures += 1
                break
            if len(placement.scale) != 3:
                print(f"❌ FAILED: bad scale length for {placement.role}")
                failures += 1
                break

    print("\n" + "=" * 70)
    if failures:
        print(f"Completed with {failures} failure(s)")
        return 1

    print("Completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
