#!/usr/bin/env python3
"""Phase 4 test: Enhanced validator with parent-cycle detection and materials support."""

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


def test_basic_scenes() -> int:
    """Test that basic scenes still pass enhanced validation."""
    test_commands = [
        "show a hydrogen atom",
        "create a solar system with 8 planets",
        "display a DNA double helix",
    ]

    failures = 0
    for command in test_commands:
        plan_obj = plan(command)
        if plan_obj is None:
            print(f"✗ planner failed for: {command}")
            failures += 1
            continue
        param = generate_parametric_scene(plan_obj)
        scene = build_scene_json(plan_obj, param, validate=True)
        try:
            validate_member1(scene)
            print(f"✓ enhanced validator: {command}")
        except Exception as e:
            print(f"✗ enhanced validator FAILED for {command}: {e}")
            failures += 1

    return failures


def test_parent_cycle_detection() -> int:
    """Test that parent cycle detection works."""
    failures = 0

    # Valid parent hierarchy (no cycle)
    valid_scene = {
        "objects": [
            {
                "id": "sun",
                "type": "sphere",
                "position": [0.0, 0.0, 0.0],
                "color": [1.0, 1.0, 0.0],
                "animation": "none",
                "orbit_center": [0.0, 0.0, 0.0],
                "orbit_speed": 0.0,
                "parent": None,
            },
            {
                "id": "planet",
                "type": "sphere",
                "position": [5.0, 0.0, 0.0],
                "color": [0.2, 0.5, 1.0],
                "animation": "orbit",
                "orbit_center": [0.0, 0.0, 0.0],
                "orbit_speed": 0.1,
                "parent": "sun",
            },
        ]
    }

    try:
        validate_member1(valid_scene)
        print("✓ parent validation: valid hierarchy accepted")
    except Exception as e:
        print(f"✗ parent validation FAILED: {e}")
        failures += 1

    # Invalid parent: cycle (self-parent)
    cycle_scene = {
        "objects": [
            {
                "id": "obj",
                "type": "sphere",
                "position": [0.0, 0.0, 0.0],
                "color": [1.0, 0.0, 0.0],
                "animation": "none",
                "orbit_center": [0.0, 0.0, 0.0],
                "orbit_speed": 0.0,
                "parent": "obj",  # self-cycle
            }
        ]
    }

    try:
        validate_member1(cycle_scene)
        print("✗ cycle detection FAILED: self-cycle not detected")
        failures += 1
    except ValueError as e:
        if "cycle" in str(e).lower():
            print(f"✓ cycle detection: self-cycle detected")
        else:
            print(f"✗ cycle detection raised wrong error: {e}")
            failures += 1

    # Invalid parent: reference to non-existent object
    bad_parent_scene = {
        "objects": [
            {
                "id": "moon",
                "type": "sphere",
                "position": [5.5, 0.0, 0.0],
                "color": [0.5, 0.5, 0.5],
                "animation": "none",
                "orbit_center": [0.0, 0.0, 0.0],
                "orbit_speed": 0.0,
                "parent": "nonexistent_planet",  # bad reference
            }
        ]
    }

    try:
        validate_member1(bad_parent_scene)
        print("✗ parent reference validation FAILED: missing parent not caught")
        failures += 1
    except ValueError as e:
        if "does not exist" in str(e).lower():
            print(f"✓ parent reference validation: missing parent detected")
        else:
            print(f"✗ parent reference validation raised wrong error: {e}")
            failures += 1

    return failures


def test_extended_fields() -> int:
    """Test validation of scale, material, and other optional fields."""
    failures = 0

    # Valid scene with material and scale
    scene_with_extras = {
        "objects": [
            {
                "id": "metallic_sphere",
                "type": "sphere",
                "position": [0.0, 0.0, 0.0],
                "color": [0.8, 0.8, 0.8],
                "animation": "none",
                "orbit_center": [0.0, 0.0, 0.0],
                "orbit_speed": 0.0,
                "material": "metallic",
                "scale": [1.5, 1.5, 1.5],
            }
        ]
    }

    try:
        validate_member1(scene_with_extras)
        print("✓ extended fields: material and scale validated")
    except Exception as e:
        print(f"✗ extended fields FAILED: {e}")
        failures += 1

    # Invalid scale (non-positive)
    bad_scale_scene = {
        "objects": [
            {
                "id": "bad_scale",
                "type": "sphere",
                "position": [0.0, 0.0, 0.0],
                "color": [1.0, 0.0, 0.0],
                "animation": "none",
                "orbit_center": [0.0, 0.0, 0.0],
                "orbit_speed": 0.0,
                "scale": [0.0, 1.0, 1.0],  # zero scale is invalid
            }
        ]
    }

    try:
        validate_member1(bad_scale_scene)
        print("✗ scale validation FAILED: zero scale not caught")
        failures += 1
    except ValueError as e:
        if "positive" in str(e).lower():
            print(f"✓ scale validation: zero scale detected")
        else:
            print(f"✗ scale validation raised wrong error: {e}")
            failures += 1

    return failures


def main() -> int:
    print("=" * 70)
    print("Phase 4 Enhanced Validator Test")
    print("=" * 70)

    total_failures = 0

    print("\n[Test 1] Basic scenes with enhanced validator")
    print("-" * 70)
    total_failures += test_basic_scenes()

    print("\n[Test 2] Parent cycle detection")
    print("-" * 70)
    total_failures += test_parent_cycle_detection()

    print("\n[Test 3] Extended fields (material, scale, metadata)")
    print("-" * 70)
    total_failures += test_extended_fields()

    print("\n" + "=" * 70)
    if total_failures:
        print(f"Completed with {total_failures} failure(s)")
        return 1

    print("All Phase 4 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
