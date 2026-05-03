#!/usr/bin/env python3
"""Quick test of the Phase 1 planner stage."""

import sys
import json

# Ensure we can import from the project root
sys.path.insert(0, "/".join(__file__.split("/")[:-2]))

from llm.planner import plan


def test_planner():
    """Test the planner on a few sample commands."""
    
    test_commands = [
        "show a hydrogen atom",
        "create a solar system with 8 planets",
        "display a DNA double helix",
        "show a water molecule"
    ]

    print("=" * 70)
    print("Phase 1 Planner Test")
    print("=" * 70)

    for command in test_commands:
        print(f"\nCommand: {command}")
        print("-" * 70)
        
        try:
            plan_obj = plan(command)
            
            if plan_obj is None:
                print("❌ FAILED: Planner returned None")
                continue
            
            print(f"✓ Scene Type: {plan_obj.scene_type}")
            print(f"✓ Description: {plan_obj.description}")
            print(f"✓ Num Objects: {plan_obj.num_objects}")
            print(f"✓ Components: {', '.join(plan_obj.components)}")
            print(f"✓ Animation Types: {', '.join(plan_obj.animation_types)}")
            print(f"✓ Hierarchy Needed: {plan_obj.hierarchy_needed}")
            print(f"✓ Use Mesh: {plan_obj.use_mesh}")
            print(f"✓ Complexity: {plan_obj.complexity}")
            
            # Print as JSON for full validation
            plan_json = json.dumps(plan_obj.model_dump(), indent=2)
            print(f"\nFull plan JSON:\n{plan_json}")
            
        except Exception as e:
            print(f"❌ FAILED: {e}")


if __name__ == "__main__":
    test_planner()
