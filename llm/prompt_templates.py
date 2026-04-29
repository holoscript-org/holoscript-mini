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


def build_planner_prompt(command: str) -> str:
    """Build prompt for the planner stage (Stage 1 of Member 1 pipeline).
    
    The planner extracts scene intent WITHOUT generating final JSON.
    Output is a structured plan: scene_type, num_objects, components, animations, etc.
    """

    planner_examples = [
        {
            "command": "show a hydrogen atom",
            "plan": {
                "scene_type": "atom",
                "description": "A hydrogen atom with a central proton and an orbiting electron",
                "num_objects": 3,
                "components": ["proton", "electron", "orbital_ring"],
                "repeat_counts": {},
                "animation_types": ["none", "orbit"],
                "hierarchy_needed": False,
                "use_mesh": False,
                "complexity": "low"
            }
        },
        {
            "command": "create a solar system with 8 planets",
            "plan": {
                "scene_type": "solar_system",
                "description": "A solar system with the sun and 8 orbiting planets, some with moons",
                "num_objects": 11,
                "components": ["sun", "planet", "moon"],
                "repeat_counts": {"planet": 8, "moon": 2},
                "animation_types": ["none", "orbit"],
                "hierarchy_needed": True,
                "use_mesh": False,
                "complexity": "medium"
            }
        },
        {
            "command": "show a DNA double helix",
            "plan": {
                "scene_type": "organic",
                "description": "A DNA double helix with intertwined strands and base pair connectors",
                "num_objects": 15,
                "components": ["strand_left", "strand_right", "base_pair"],
                "repeat_counts": {"base_pair": 12},
                "animation_types": ["none"],
                "hierarchy_needed": False,
                "use_mesh": False,
                "complexity": "medium"
            }
        },
        {
            "command": "display a human heart",
            "plan": {
                "scene_type": "organic",
                "description": "A realistic human heart with chambers and detailed internal structure",
                "num_objects": 1,
                "components": ["heart_mesh"],
                "repeat_counts": {},
                "animation_types": ["none"],
                "hierarchy_needed": False,
                "use_mesh": True,
                "complexity": "high"
            }
        }
    ]

    examples_str = "\n\n".join(
        f"Command: \"{ex['command']}\"\nPlan: {json.dumps(ex['plan'], indent=2)}"
        for ex in planner_examples
    )

    return f"""You are a 3D scene planner. Extract the abstract intent from a user command.

Do NOT generate final scene JSON. Instead, produce a plan that describes what the scene should contain:

Output format (raw JSON, no markdown, no explanation):
{{
  "scene_type": "atom|molecule|solar_system|mechanical|organic|abstract|geometric|crystalline|astronomical",
  "description": "plain language summary",
  "num_objects": <1-20>,
  "components": ["list", "of", "major", "building", "blocks"],
    "repeat_counts": {{"component_name": <count>}},
  "animation_types": ["none", "orbit", "spin"],
  "hierarchy_needed": true/false,
  "use_mesh": true/false,
  "complexity": "low|medium|high"
}}

Rules:
- scene_type: What kind of scene? (atom, molecule, solar system, etc.)
- description: What does the user want? Plain language.
- num_objects: How many objects total? (estimate is fine, 1-20)
- components: What are the major parts? Use unique role names only. Do not repeat names in the list.
- repeat_counts: Use this for repeated elements. Example: {{"planet": 8, "moon": 2}}
- animation_types: Which animation types? (none, orbit, spin)
- hierarchy_needed: Do objects have parent-child relationships? (e.g., moon orbits earth)
- use_mesh: Should any part use a 3D mesh model instead of primitives? (organic shapes need meshes)
- complexity: low (<5 objects), medium (6-15), high (16-20)

Stability rules:
- Keep component names deterministic and role-based.
- Prefer {{"left", "right", "center", "ring"}} style names for symmetrical structures.
- When a structure repeats, encode the repetition in repeat_counts instead of duplicating component names.
- The planner should be general-purpose: it must work for atoms, molecules, solar systems, structures, vehicles, characters, landscapes, diagrams, and abstract scenes.

Examples:

{examples_str}

User command: {command}

Output only the JSON plan. Nothing else."""
