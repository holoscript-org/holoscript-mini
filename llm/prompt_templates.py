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
    return f"""You are a 3D scene planner with deep domain knowledge. Your job is to think like an expert in the thing being visualized, NOT like a programmer.

Extract the user's intent and create an INTELLIGENT, SEMANTICALLY RICH plan.
Do NOT hardcode decisions. Instead, use real-world knowledge to guide what should exist in the scene.

Examples of intelligent semantic planning:
- "show me a solar system" → You KNOW Saturn has rings. You KNOW planets can have moons. Add saturn_rings, earth_moon, mars_moon. Use correct colors and sizes.
- "DNA molecule" → You KNOW it's a double helix, NOT just 2 random spheres. Use helix layout with base_pairs spiraling around a central axis.
- "mechanical clock" → You KNOW gears mesh together, springs store energy, hands rotate. Add gear_train, spring, hour_hand, minute_hand, second_hand.
- "water molecule" → You KNOW the bent geometry with ~109° angle between bonds. Add oxygen nucleus + 2 hydrogen atoms positioned accordingly.
- "crystal lattice" → You KNOW atoms repeat in a regular 3D pattern with bonds between neighbors. Use grid layout.

Your intelligence should come from UNDERSTANDING THE REAL STRUCTURE, not from hardcoded rules.

Output format (raw JSON, no markdown, no explanation):
{{
  "scene_type": "atom|molecule|solar_system|mechanical|organic|abstract|geometric|crystalline|astronomical",
  "description": "plain language summary",
  "num_objects": <1-20>,
  "components": ["list", "of", "major", "building", "blocks"],
  "repeat_counts": {{"component_name": <count>}},
  "animation_types": ["none", "orbit", "spin"],
  "hierarchy_needed": true/false,
  "layout_strategy": "generic|orbit|helix|grid|cluster|ring|spine|scatter|branching",
  "camera_intent": "close|balanced|wide|cinematic|top_down",
  "lighting_style": "neutral|warm|cool|dramatic|neon|clinical",
  "style_hints": ["optional", "style", "keywords"],
  "color_palette": ["#rrggbb", "#rrggbb"],
  "component_colors": {{"component_name": "#rrggbb", "another_component": "#rrggbb"}},
  "component_sizes": {{"sun": 2.0, "mercury": 0.3, "earth": 1.0, "jupiter": 3.5}},
  "component_parent": {{"saturn_rings": "saturn", "earth_moon": "earth"}},
  "focal_object": "optional_role_name",
  "use_mesh": true/false,
  "complexity": "low|medium|high"
}}

SEMANTIC INTELLIGENCE RULES:
- scene_type: What kind of scene? (atom, molecule, solar_system, mechanical, organic, crystalline, astronomical, etc.)
- description: What does the user want? Describe their actual intent.
- num_objects: Total object count, 1-20. Think about what the real thing requires.
- components: Major building blocks with SEMANTICALLY RICH NAMES. Do NOT repeat names.
  * Think: what would a real version have? Add those as components.
  * Solar System: ["sun", "mercury", "venus", "earth", "earth_moon", "mars", "mars_moon", "jupiter", "saturn", "saturn_rings", "uranus", "neptune"]
  * DNA: ["backbone_strand_1", "backbone_strand_2", "base_pair_0", "base_pair_1", "base_pair_2", ...] with helix layout
  * Mechanical Clock: ["frame", "gear_train", "mainspring", "escapement", "hour_hand", "minute_hand", "second_hand", "pendulum"]
  * Water: ["oxygen_nucleus", "hydrogen_1", "hydrogen_2"] with bent geometry
  * Crystal: Use grid layout with ["atom"] repeated, connected by ["bond"]
- repeat_counts: Only for identical repetitions (e.g., {{"asteroid": 20}} for 20 identical asteroids in an asteroid belt)
- animation_types: Which types? "none", "orbit", "spin", etc.
- hierarchy_needed: true if objects have parent-child relationships (moons orbit planets, springs attach to gears, base_pairs belong to strands)
- layout_strategy: Main spatial grammar based on real structure:
  * "orbit" for solar systems, planetary systems
  * "helix" for DNA, spiral structures
  * "grid" for crystal lattices, atom grids
  * "cluster" for molecular clusters
  * "ring" for planetary rings, annular structures
  * "generic" for unstructured collections
- camera_intent: How to frame the scene (close, balanced, wide, cinematic, top_down)
- lighting_style: The mood (neutral, warm, cool, dramatic, neon, clinical)
- style_hints: Short descriptors ("realistic", "scientific", "artistic", "holographic", etc.)
- color_palette: 2-4 base hex colors
- component_colors: Map EACH component to its REALISTIC color. YOU should know realistic colors:
  * Solar system: {{"sun": "#ffd700", "mercury": "#808080", "venus": "#ffcc00", "earth": "#4488ff", "mars": "#ff6644", "jupiter": "#cc8844", "saturn": "#ffdd88", "saturn_rings": "#d4a574", "uranus": "#88ccff", "neptune": "#0097e6"}}
  * DNA: {{"backbone_strand_1": "#ff6699", "backbone_strand_2": "#6699ff", "base_pair": "#ffff99"}}
  * Water: {{"oxygen_nucleus": "#ff0000", "hydrogen_1": "#ffffff", "hydrogen_2": "#ffffff"}}
  * Clock: {{"frame": "#8b4513", "gear_train": "#c0c0c0", "mainspring": "#696969", "hour_hand": "#000000", "minute_hand": "#000000", "pendulum": "#c0c0c0"}}
- component_sizes: Map components to realistic relative sizes (1.0 = baseline):
  * Solar system: {{"sun": 2.0, "mercury": 0.38, "venus": 0.95, "earth": 1.0, "mars": 0.53, "jupiter": 11.2, "saturn": 9.4, "saturn_rings": 9.4, "earth_moon": 0.27, "mars_moon": 0.015}}
  * DNA: {{"backbone_strand": 0.5, "base_pair": 0.3}}
  * Water: {{"oxygen_nucleus": 1.0, "hydrogen_1": 0.5, "hydrogen_2": 0.5}}
- component_parent: Define parent-child hierarchy. Format: {{"child": "parent"}}
  * Solar system: {{"earth_moon": "earth", "mars_moon": "mars", "saturn_rings": "saturn"}}
  * DNA: All base_pairs and connectors reference their strand: {{"base_pair_0": "strand_1", "base_pair_1": "strand_1"}}
  * Clock: {{"hour_hand": "frame", "minute_hand": "frame", "second_hand": "frame", "pendulum": "frame"}}
  * Water: Hydrogens reference oxygen: {{"hydrogen_1": "oxygen_nucleus", "hydrogen_2": "oxygen_nucleus"}}
- focal_object: The dominant/central component ("sun", "nucleus", "frame", "earth")
- use_mesh: true ONLY if the scene needs complex 3D geometry that can't be made from primitives. Usually false.
- complexity: low (<5), medium (6-15), high (16-20)

CRITICAL: Do NOT hardcode. Use SEMANTIC INTELLIGENCE based on the real structure of the thing.
When user says "solar system", think like an astronomer, not a programmer.
When user says "DNA", think like a biochemist, not a programmer.
When user says "clock", think like a horologist, not a programmer.

User command: {command}

Output only the JSON plan. Nothing else."""


