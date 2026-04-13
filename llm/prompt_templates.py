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
                "color": [1.0, 0.95, 0.3],
                "secondary_color": [1.0, 0.6, 0.0],
                "size": 2.5,
                "surface_style": "emissive_glow",
                "animation": "none",
                "orbit_center": [0.0, 0.0, 0.0],
                "orbit_speed": 0.0
            },
            {
                "id": "mercury",
                "type": "sphere",
                "position": [4.0, 0.0, 0.0],
                "color": [0.68, 0.66, 0.63],
                "secondary_color": [0.5, 0.48, 0.46],
                "size": 0.25,
                "surface_style": "polar_caps",
                "animation": "orbit",
                "orbit_center": [0.0, 0.0, 0.0],
                "orbit_speed": 1.607
            },
            {
                "id": "venus",
                "type": "sphere",
                "position": [6.5, 0.0, 0.0],
                "color": [0.9, 0.78, 0.5],
                "secondary_color": [0.95, 0.88, 0.65],
                "size": 0.45,
                "surface_style": "plain",
                "animation": "orbit",
                "orbit_center": [0.0, 0.0, 0.0],
                "orbit_speed": 1.174
            },
            {
                "id": "earth",
                "type": "sphere",
                "position": [9.0, 0.0, 0.0],
                "color": [0.1, 0.35, 0.8],
                "secondary_color": [0.13, 0.5, 0.18],
                "size": 0.5,
                "surface_style": "earth",
                "animation": "orbit",
                "orbit_center": [0.0, 0.0, 0.0],
                "orbit_speed": 1.0
            },
            {
                "id": "mars",
                "type": "sphere",
                "position": [11.5, 0.0, 0.0],
                "color": [0.75, 0.32, 0.12],
                "secondary_color": [0.92, 0.92, 0.92],
                "size": 0.3,
                "surface_style": "polar_caps",
                "animation": "orbit",
                "orbit_center": [0.0, 0.0, 0.0],
                "orbit_speed": 0.802
            },
            {
                "id": "jupiter",
                "type": "sphere",
                "position": [15.5, 0.0, 0.0],
                "color": [0.78, 0.62, 0.46],
                "secondary_color": [0.62, 0.42, 0.28],
                "size": 1.2,
                "surface_style": "banded",
                "bands": [
                    {"color": [0.82, 0.7, 0.55], "width": 0.12},
                    {"color": [0.58, 0.38, 0.24], "width": 0.08},
                    {"color": [0.85, 0.73, 0.58], "width": 0.1},
                    {"color": [0.6, 0.4, 0.26], "width": 0.07},
                    {"color": [0.88, 0.76, 0.6], "width": 0.14},
                    {"color": [0.55, 0.36, 0.22], "width": 0.06},
                    {"color": [0.8, 0.68, 0.52], "width": 0.1},
                    {"color": [0.63, 0.44, 0.3], "width": 0.08},
                    {"color": [0.84, 0.72, 0.56], "width": 0.12},
                    {"color": [0.57, 0.37, 0.23], "width": 0.07},
                    {"color": [0.8, 0.68, 0.52], "width": 0.06}
                ],
                "animation": "orbit",
                "orbit_center": [0.0, 0.0, 0.0],
                "orbit_speed": 0.434
            },
            {
                "id": "saturn",
                "type": "sphere",
                "position": [20.0, 0.0, 0.0],
                "color": [0.87, 0.8, 0.58],
                "secondary_color": [0.75, 0.65, 0.42],
                "size": 1.0,
                "surface_style": "saturn_rings",
                "bands": [
                    {"color": [0.9, 0.84, 0.64], "width": 0.15},
                    {"color": [0.78, 0.7, 0.5], "width": 0.1},
                    {"color": [0.92, 0.86, 0.66], "width": 0.18},
                    {"color": [0.76, 0.68, 0.48], "width": 0.08},
                    {"color": [0.89, 0.82, 0.62], "width": 0.12}
                ],
                "ring": {
                    "inner_radius_factor": 1.3,
                    "outer_radius_factor": 2.4,
                    "color": [0.8, 0.72, 0.55],
                    "opacity": 0.75
                },
                "animation": "orbit",
                "orbit_center": [0.0, 0.0, 0.0],
                "orbit_speed": 0.323
            },
            {
                "id": "uranus",
                "type": "sphere",
                "position": [24.0, 0.0, 0.0],
                "color": [0.53, 0.82, 0.88],
                "secondary_color": [0.44, 0.74, 0.82],
                "size": 0.7,
                "surface_style": "banded",
                "bands": [
                    {"color": [0.56, 0.84, 0.9], "width": 0.2},
                    {"color": [0.47, 0.77, 0.84], "width": 0.15},
                    {"color": [0.58, 0.85, 0.91], "width": 0.18},
                    {"color": [0.45, 0.75, 0.82], "width": 0.12},
                    {"color": [0.55, 0.83, 0.89], "width": 0.2}
                ],
                "animation": "orbit",
                "orbit_center": [0.0, 0.0, 0.0],
                "orbit_speed": 0.228
            },
            {
                "id": "neptune",
                "type": "sphere",
                "position": [27.5, 0.0, 0.0],
                "color": [0.14, 0.28, 0.9],
                "secondary_color": [0.2, 0.4, 0.95],
                "size": 0.65,
                "surface_style": "banded",
                "bands": [
                    {"color": [0.18, 0.33, 0.92], "width": 0.18},
                    {"color": [0.1, 0.22, 0.85], "width": 0.12},
                    {"color": [0.22, 0.38, 0.94], "width": 0.15},
                    {"color": [0.12, 0.25, 0.87], "width": 0.1},
                    {"color": [0.19, 0.35, 0.93], "width": 0.18}
                ],
                "animation": "orbit",
                "orbit_center": [0.0, 0.0, 0.0],
                "orbit_speed": 0.182
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
- "secondary_color" is optional and, if present, must be a list of exactly 3 floats
- "size" is optional and, if present, must be a positive float
- "surface_style" is optional and may be: plain, emissive_glow, polar_caps, earth, banded, saturn_rings
- "bands" is optional and, if present, must be a list of band objects with color and width
- "ring" is optional and, if present, must contain inner_radius_factor, outer_radius_factor, color, and opacity
- Minimum 1 object. Maximum 20 objects.
- Required fields are: id, type, position, color, animation, orbit_center, orbit_speed.
- Preserve richer fields already present when refining a scene.

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
