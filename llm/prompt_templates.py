import json
from core.state.scene_grammar import SCENE_GRAMMAR


def build_system_prompt() -> str:
    schema_str = json.dumps(SCENE_GRAMMAR, indent=2)

    few_shot_solar = json.dumps({
        "objects": [
            {
                "id": "sun",
                "type": "sphere",
                "position": [0.0, 0.0, 0.0],
                "color": [1.0, 0.84, 0.0],
                "animation": "none",
                "orbit_center": [0.0, 0.0, 0.0],
                "orbit_speed": 0.0
            },
            {
                "id": "earth",
                "type": "sphere",
                "position": [3.0, 0.0, 0.0],
                "color": [0.0, 0.4, 1.0],
                "animation": "orbit",
                "orbit_center": [0.0, 0.0, 0.0],
                "orbit_speed": 0.5
            },
            {
                "id": "mars",
                "type": "sphere",
                "position": [5.0, 0.0, 0.0],
                "color": [0.8, 0.3, 0.1],
                "animation": "orbit",
                "orbit_center": [0.0, 0.0, 0.0],
                "orbit_speed": 0.3
            }
        ]
    }, indent=2)

    few_shot_water = json.dumps({
        "objects": [
            {
                "id": "oxygen",
                "type": "sphere",
                "position": [0.0, 0.0, 0.0],
                "color": [1.0, 0.0, 0.0],
                "animation": "none",
                "orbit_center": [0.0, 0.0, 0.0],
                "orbit_speed": 0.0
            },
            {
                "id": "hydrogen1",
                "type": "sphere",
                "position": [1.2, 0.9, 0.0],
                "color": [0.9, 0.9, 0.9],
                "animation": "none",
                "orbit_center": [0.0, 0.0, 0.0],
                "orbit_speed": 0.0
            },
            {
                "id": "hydrogen2",
                "type": "sphere",
                "position": [-1.2, 0.9, 0.0],
                "color": [0.9, 0.9, 0.9],
                "animation": "none",
                "orbit_center": [0.0, 0.0, 0.0],
                "orbit_speed": 0.0
            }
        ]
    }, indent=2)

    few_shot_abstract = json.dumps({
        "objects": [
            {
                "id": "core_cube",
                "type": "cube",
                "position": [0.0, 0.0, 0.0],
                "color": [0.5, 0.0, 1.0],
                "animation": "none",
                "orbit_center": [0.0, 0.0, 0.0],
                "orbit_speed": 0.0
            },
            {
                "id": "outer_ring",
                "type": "ring",
                "position": [0.0, 0.0, 0.0],
                "color": [0.0, 1.0, 1.0],
                "animation": "orbit",
                "orbit_center": [0.0, 0.0, 0.0],
                "orbit_speed": 1.2
            },
            {
                "id": "accent_cylinder",
                "type": "cylinder",
                "position": [2.0, 1.0, 0.0],
                "color": [1.0, 0.5, 0.0],
                "animation": "none",
                "orbit_center": [0.0, 0.0, 0.0],
                "orbit_speed": 0.0
            }
        ]
    }, indent=2)

    return f"""You are a 3D scene generator. Output ONLY raw JSON. No markdown. No explanations. No code blocks.

The JSON must conform exactly to this schema:
{schema_str}

Rules:
- "type" must be one of: sphere, cube, cylinder, ring, label
- "animation" must be one of: none, orbit
- "position", "color", "orbit_center" must each be a list of exactly 3 floats
- "color" floats are in range 0.0 to 1.0
- Minimum 1 object. Maximum 20 objects.
- All fields are required for every object.

Examples:

Example 1 - Solar system:
{few_shot_solar}

Example 2 - Water molecule:
{few_shot_water}

Example 3 - Abstract geometry:
{few_shot_abstract}

Output only the JSON object. Nothing else."""


def build_refinement_prompt(previous_scene: dict, new_command: str) -> str:
    previous_str = json.dumps(previous_scene, indent=2)
    return f"""You are modifying an existing 3D scene based on a user command.

Current scene:
{previous_str}

User command: {new_command}

Apply the command to the scene. Output ONLY the modified scene as raw JSON. No markdown. No explanations.
The output must conform to the same schema as the current scene."""
