"""
Scene architect: builds the scene JSON via three sequential LLM passes
instead of one monolithic call.

  7a. Layout & Composition  (_architect_layout)  — decides WHAT exists and
      roughly WHERE: object ids, primitive-vs-mesh type, geometry family,
      coarse position, parent/child skeleton, labels. No materials, no fine
      geometry numbers, no animation, no lights/camera yet.
  7b. Per-Object Detail     (_architect_detail)  — given the layout skeleton,
      fills in full geometry params, materials, animation (including
      physics), and educational label/description text. Explicitly NOT
      allowed to add/remove/reposition objects wholesale — only refine.
  7c. Lighting/Camera/Polish (_architect_finish) — given the fully-detailed
      object list, decides lights, camera, and the scene's name/summary.
      Cheaper model (Flash) since this is a bounded, mechanical decision
      given full context, not open-ended composition.

Why split a single prompt into three: with everything in one call, a model
has to simultaneously juggle scene composition, per-object numeric precision
(colors, radii, physics constants), and overall narrative coherence — and
quality degrades as scene complexity grows. Splitting means each call has a
narrower, more supervisable job, and each pass maps onto a natural seam that
already existed in the single prompt's constants (_RULES' primitive/mesh +
scale sections -> layout; _SCHEMA + _EDUCATION + _PHYSICS + few-shots ->
detail; _RULES' lighting/camera sections -> finish).

Each pass takes an `intent_ir` argument — the Scene Intent IR produced by
pipeline/intent_extractor.py (Stage 3) — as additional advisory context. This
is never validated or written to the final scene; it's a structured hint of
what the user asked for (objects, spatial relationships, dynamics, mood).

generate_scene() orchestrates 7a -> 7b -> 7c. If 7a fails (no object
skeleton at all), the whole architect stage reports failure and the caller
(pipeline/pipeline_runner.py) falls through to the legacy retrieval/builder
path — same failure semantics as before this rebuild. If 7b or 7c fail
individually, _default_lighting_camera() provides a deterministic safety net
(hardcoded 3-light rig + bounding-box camera) rather than aborting the scene
entirely.

Models: Gemini via Vertex AI (ADC, no API key) is the primary path for every
pass; Groq Llama 3.3 70B is the fallback. Both are reached through the shared
client in llm/gemini_client.py — this module no longer constructs its own
Vertex AI client or makes its own Groq REST call.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from core.utils.logger import get_logger
from llm.gemini_client import call_llm
from pipeline.events import OnEvent, make_emitter, COMPLETED, OUTPUT, STARTED

logger = get_logger("scene_architect")

_MODEL_GEMINI_PRO = "gemini-2.5-pro"
_MODEL_GEMINI_FLASH = "gemini-2.5-flash"
_MODEL_GROQ = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Schema block — mirrors scene_validator.py exactly
# ---------------------------------------------------------------------------

_SCHEMA = """\
SCHEMA (follow exactly — validator rejects any deviation):

Object (primitive):
  { "id": "snake_case_unique", "type": "primitive",
    "geometry": { "type": "sphere|box|cylinder|ring|torus|capsule|plane",
      sphere  → radius (number > 0)
      box     → width, height, depth
      cylinder→ radius, from:[x,y,z], to:[x,y,z]   ← use this for tubes, vessels, arms
      ring    → innerRadius, outerRadius, thetaSegments (≥3, default 128 for smooth)
      torus   → radius, tube
      capsule → radius, length
      plane   → width, height },
    "position": [x,y,z],
    "scale": [sx,sy,sz],        ← use non-uniform scale for ellipsoids / stretched shapes
    "rotation": [rx,ry,rz],     ← euler degrees, optional
    "material": { "type":"standard", "color":"#rrggbb", "roughness":0-1, "metalness":0-1,
      optional: "emissive":"#rrggbb", "emissiveIntensity":0-5,
                "opacity":0-1, "transparent":true },
    "animation": { "type":"none|orbit|spin|physics",
      orbit   → "center":[x,y,z], "speed": float, "phase": float (radians, vary per planet)
      spin    → "axis":[x,y,z],   "speed": float
      physics → see PHYSICS ANIMATION section below },
    "parent": "other_id",       ← optional; child position is relative to parent
    "label": "Human-readable",  ← optional, show for key objects only
    "description": "..."        ← optional; 1-2 sentence educational explanation of
                                   this object's role in the phenomenon (see EDUCATIONAL
                                   OUTPUT below). Provide for every labelled object.
  }

Object (mesh — ONLY if path appears in AVAILABLE MESHES below):
  { "id": "...", "type": "mesh", "model": "/assets/meshes/...",
    "position": [...], "scale": [...],
    "material": { "type":"standard", "color":"#ffffff", "roughness":0.5, "metalness":0.0 },
    "animation": {...}, "label": "...", "description": "..." }
  IMPORTANT: For mesh objects always use color "#ffffff" — the GLB has embedded textures;
  any other color will tint/destroy the original appearance.
  MESH SCALE: All GLB models are auto-normalized to a 2-unit bounding box at load time.
  So scale [1,1,1] = 1 unit half-size (2 units wide). Use scale relative to that:
    Single subject (ball, figure, animal): scale [3,3,3] to [5,5,5]
    Wide floor/ground:  scale [8,1,8] to [12,1,12]
    Tall structure:     scale [3,6,3]
  For physics gravity: floor_y = floor_position_y + (object_scale / 2)
    Example: floor at y=0, basketball scale [0.5,0.5,0.5] → floor_y = 0 + 0.25 = 0.25
    Starting y for drop: floor_y + 8 to floor_y + 14 for a visible long drop

Lights:
  { "type":"ambient",     "intensity":0-2,  "color":"#rrggbb" }
  { "type":"directional", "intensity":0-3,  "color":"#rrggbb", "position":[x,y,z], "castShadow":true }
  { "type":"point",       "intensity":0-10, "color":"#rrggbb", "position":[x,y,z] }

Camera: { "position":[x,y,z], "target":[x,y,z], "fov":40-75 }

Scene root (top level):
  { "name": "Short Title",     ← 2-5 words, e.g. "Basketball Bouncing on Hardwood"
    "summary": "...",          ← REQUIRED. Educational explanation (see below).
    "objects": [...], "lights": [...], "camera": {...} }\
"""

# ---------------------------------------------------------------------------
# Educational narration block
# ---------------------------------------------------------------------------

_EDUCATION = """\
═══ EDUCATIONAL OUTPUT ═══
This system is used for teaching. Every scene MUST explain the concept it depicts.

Scene-level "summary" (REQUIRED, top-level string, 2-4 sentences):
  • Name the actual principle, law, or system being shown — not what the scene looks like.
  • Explain WHY the scene behaves the way it does, referencing the real numbers you chose
    (g value, restitution, orbital speeds, radii, amplitude, etc.).
  • Write for a curious student: plain language, no jargon without a short gloss.
  • Never describe the rendering ("a red sphere sits at the centre") — describe the
    phenomenon ("the Sun holds the planets in orbit through gravitational attraction").

  GOOD: "This scene demonstrates gravity and inelastic collision. The basketball falls
         under Earth's gravitational acceleration (g = 9.8 m/s²) and retains 70% of its
         energy on each bounce (restitution 0.7), so each rebound is noticeably lower
         than the last until the ball settles on the floor."
  BAD:  "A basketball and a wooden floor are shown in the scene."

Object-level "description" (1-2 sentences, on every object that has a "label"):
  • Explain that object's specific role in the phenomenon.
  • Where a number was chosen for a reason, say the reason.

  GOOD: "Earth completes one orbit per unit of simulation time and sits 9 units from the
         Sun — the reference distance all other orbital radii here are scaled against."
  BAD:  "This is the Earth."

Labelling rule: give a "label" + "description" to every object a student would ask about.
For a single imported mesh (e.g. one skeleton or one organ), the whole object gets one
label and one description — do NOT invent labels for parts you cannot address separately.\
"""

# ---------------------------------------------------------------------------
# Rules block, split along the layout / lighting seam.
# ---------------------------------------------------------------------------

_RULES_LAYOUT = """\
═══ PRIMITIVE vs MESH ═══
HIGHEST PRIORITY RULE — MESHES OVERRIDE EVERYTHING ELSE:
  If a concept appears in AVAILABLE MESHES (shown in the prompt), you MUST use
  type="mesh" for that concept. Do NOT use a primitive. The file is confirmed on disk.
  This rule overrides all category rules below. No exceptions.

Use primitives for EVERYTHING ELSE, including:
  • ALL astronomical: stars, planets, moons, rings, asteroid belts, galaxies, nebulae
  • ALL atomic / molecular: nucleus, protons, neutrons, electrons, bonds, shells
  • ALL mechanical/structural: gears, orreries, clock faces, crystal lattices,
    bridges, towers, molecules, grids
  • ANY abstract concept, energy field, wave, or geometric arrangement
  • NEVER invent a mesh path — only use paths listed in AVAILABLE MESHES

═══ SCALE REFERENCE (for coarse placement — refine exact radii in detail pass) ═══
Astronomical scale reference:
  Sun r=2.5, Jupiter r=1.2, Saturn r=1.0, Uranus r=0.7, Neptune r=0.65,
  Earth r=0.5, Venus r=0.45, Mars r=0.3, Mercury r=0.25, Moon r=0.14
  Orbital radii: Mercury 4, Venus 6.5, Earth 9, Mars 11.5, Jupiter 16, Saturn 21

Atomic scale reference:
  Nucleus r=0.4-0.8, electron shells at r=3/5/7, electrons r=0.1-0.15
  Proton/neutron r=0.2 packed inside nucleus zone

Molecular scale reference:
  Oxygen r=0.5, Hydrogen r=0.3, bond cylinder radius=0.1 between atom centers
  Water: O at [0,0,0], H1 at [0.754, -0.588, 0], H2 at [-0.754, -0.588, 0]

Parent-child usage:
  • Moon → parent: "earth"  (position relative to earth)
  • Saturn ring → parent: "saturn"
  • Electron → parent: none (use orbit animation with center=[0,0,0] for shells)
  • Attached mechanical arms → parent: "base"

CRITICAL — PRE-COMPUTED NUMBERS ONLY:
  ALL numeric values (positions, radii) MUST be pre-computed decimal numbers.
  NEVER write math expressions, formulas, trig calls, or code of any kind.\
"""

_RULES_FINISH = """\
═══ LIGHTING ═══
Space / astronomical scenes:
  ambient 0.08-0.18 (dark space) + point light at star [0,0,0] intensity 4-8 warm color
  + 1 directional for rim lighting
  Star sphere: emissive="#ff8800" or "#ffcc44", emissiveIntensity 0.9-1.2

Biological / anatomical:
  ambient 0.35-0.45 warm + directional key light + directional fill (opposite side)
  + subtle blue-tinted point for depth

Mechanical / structural:
  ambient 0.3 + directional 1.0-1.4 + optional cool point for metallic sheen

Abstract / artistic:
  ambient 0.4 + directional 1.0 + colored point lights for atmosphere

═══ CAMERA ═══
Camera positioning:
  • Find the bounding box of all objects
  • scene_radius = max(all object distances from center)
  • camera.position = [cx, cy + scene_radius*0.6, cz + scene_radius*2.0]
  • camera.target = scene center
  • fov: 60-65 for orbital scenes, 45-55 for anatomical close-up, 55-65 general
  • Never place camera inside the scene or behind objects

CRITICAL — PRE-COMPUTED NUMBERS ONLY: same rule as elsewhere — write only final
decimal numbers, never expressions or code.\
"""

# ---------------------------------------------------------------------------
# Inline quality examples (few-shot) — used by the detail pass
# ---------------------------------------------------------------------------

_EXAMPLE_ORBITAL = """\
EXAMPLE — "solar system" (orbital primitive scene, use as quality reference):
{
  "name": "Solar System",
  "summary": "This scene shows how gravity organises a planetary system. The Sun holds every planet in orbit through gravitational attraction, and because that pull weakens with distance, planets further out travel more slowly — Saturn at 21 units completes an orbit roughly three times slower than Earth at 9 units. Sizes and orbital radii here are scaled down relative to each other so the whole system stays visible at once.",
  "camera": {"position":[0,18,55], "target":[0,0,0], "fov":60},
  "lights": [
    {"type":"ambient","intensity":0.12,"color":"#ffffff"},
    {"type":"point","intensity":5.0,"color":"#ffcc66","position":[0,0,0]},
    {"type":"directional","intensity":0.5,"color":"#ffffff","position":[50,30,20],"castShadow":true}
  ],
  "objects": [
    {"id":"sun","type":"primitive","geometry":{"type":"sphere","radius":2.5},"position":[0,0,0],"scale":[1,1,1],"material":{"type":"standard","color":"#ffdd33","roughness":0.4,"metalness":0.0,"emissive":"#ff8800","emissiveIntensity":1.0},"label":"Sun","description":"The star at the centre of the system. It holds 99.8% of the system's mass, and that mass is the source of the gravitational pull keeping every planet in orbit.","animation":{"type":"none"}},
    {"id":"earth","type":"primitive","geometry":{"type":"sphere","radius":0.5},"position":[9,0,0],"scale":[1,1,1],"material":{"type":"standard","color":"#1a5acc","roughness":0.7,"metalness":0.05},"label":"Earth","description":"Orbits at 9 units from the Sun — the reference distance every other orbital radius here is scaled against. Its speed of 1.0 sets the baseline all other orbital speeds are measured relative to.","animation":{"type":"orbit","center":[0,0,0],"speed":1.0,"phase":0}},
    {"id":"moon","type":"primitive","geometry":{"type":"sphere","radius":0.14},"position":[0.65,0,0],"scale":[1,1,1],"parent":"earth","material":{"type":"standard","color":"#cccccc","roughness":0.95,"metalness":0.0},"animation":{"type":"orbit","center":[0,0,0],"speed":13.0}},
    {"id":"saturn","type":"primitive","geometry":{"type":"sphere","radius":1.0},"position":[21,0,0],"scale":[1,1,1],"material":{"type":"standard","color":"#ddcc88","roughness":0.5,"metalness":0.05},"label":"Saturn","description":"At 21 units out, Saturn orbits at 0.323 — about a third of Earth's speed. This slowdown with distance is Kepler's third law in action: the further a planet sits from the Sun, the weaker the pull and the longer its year.","animation":{"type":"orbit","center":[0,0,0],"speed":0.323,"phase":1.2}},
    {"id":"saturn_ring","type":"primitive","geometry":{"type":"ring","innerRadius":1.35,"outerRadius":2.35,"thetaSegments":128},"position":[0,0,0],"rotation":[16,0,8],"scale":[1,1,1],"parent":"saturn","material":{"type":"standard","color":"#ccbb77","roughness":0.8,"metalness":0.0,"opacity":0.6,"transparent":true},"animation":{"type":"none"}}
  ]
}
END EXAMPLE\
"""

_EXAMPLE_ANATOMICAL = """\
EXAMPLE — "human heart" (anatomical primitive scene, compound shapes):
  summary: "This model shows the four-chambered structure of the human heart and how blood
            is routed through it. The left ventricle is built largest because it pumps
            oxygenated blood to the entire body and needs the thickest muscle wall, while
            the right ventricle only pushes blood as far as the lungs. The translucent outer
            layer is the pericardium, the protective sac enclosing the whole organ."
Key objects (abbreviated — full scene uses ~25 primitives, each labelled one carries a description):
  left_ventricle_core: sphere r=1.55, scale=[0.95,1.34,1.06], color=#8f1a1a, pos=[-0.62,-1.35,0.12]
  right_ventricle_core: sphere r=1.24, color=#6f1014
  aorta_root: cylinder from=[-0.35,1.02,0.08] to=[-0.35,2.22,0.1] r=0.38, color=#c23434
  pericardium: sphere r=2.48, opacity=0.18, transparent=true (outer membrane)
  Camera: position=[0.2,1.4,11.8], target=[0,-0.4,0.2], fov=46
  Lights: warm ambient 0.38 + 2 directionals (key + fill) + subtle blue point
END EXAMPLE\
"""

# ---------------------------------------------------------------------------
# Physics animation prompt block — used by the detail pass
# ---------------------------------------------------------------------------

_PHYSICS = """\
═══ PHYSICS ANIMATION ═══
Use animation.type = "physics" when the user describes motion that follows natural laws.
The LLM decides which physics_type fits best — these are never triggered by keywords alone.

When to use each physics_type:
  gravity    → falling, dropping, bouncing, ball dropping, rain, avalanche
  pendulum   → swinging, pendulum clock, hanging weight, grandfather clock, wrecking ball
  shm        → oscillating, vibrating, spring, bobbing, pulsing, wave motion
  projectile → thrown, launched, fired, arc, ballistic, cannon, trajectory

Schema:
  {
    "animation": {
      "type": "physics",
      "physics_type": "gravity|shm|pendulum|projectile",

      // gravity / projectile
      "g": 9.8,            // gravitational acceleration:
                           //   9.8=earth, 1.6=moon, 3.7=mars, 24.8=jupiter, 0.6=asteroid, 0=weightless
      "floor_y": -2.0,     // MUST be strictly below the object's starting y position
      "restitution": 0.7,  // bounce energy retention: 0=dead stop, 1=perfect bounce

      // shm only
      "axis": "y",         // oscillation axis: "x", "y", or "z"
      "amplitude": 1.5,    // max displacement from rest position (keep ≤ 30% of scene size)
      "frequency": 0.5,    // cycles per second

      // pendulum only
      "pivot": [0, 5, 0],  // MUST have pivot.y > object's starting y position
      "arm_length": 5.0,   // hint: the renderer derives arm from position–pivot distance
      "amplitude": 0.785,  // max swing angle in RADIANS (π/4 ≈ 0.785, π/6 ≈ 0.524)
      "frequency": 0.4,    // cycles per second

      // projectile only
      "initial_velocity": [2, 8, 0],   // [vx, vy, vz] in units/sec

      // all types
      "damping": 0.02      // decay coefficient: 0=perpetual, 0.05=slow decay, 0.5=fast decay
    }
  }

Critical rules:
  • g MUST reflect the environment: moon/space/asteroid → g ≤ 2.0, NOT 9.8
  • Starting y for gravity/projectile objects: keep between 1.5 and 3.0 so the object
    is visible in frame from the first frame. NEVER start above y=4.
  • floor_y MUST always be strictly below the object's starting y position
    Example: object at y=2 → floor_y must be < 2, e.g. 0 or 0.25
  • amplitude for shm: keep ≤ 30% of the scene's bounding box size
  • pendulum pivot MUST have y > object starting y (pivot is above the bob)
    Place the bob (object position) directly below the pivot: same x and z
    Example: object at [3, 0, 0], pivot at [3, 5, 0], arm_length = 5
  • For gravity/projectile scenes: add a ground plane object at floor_y
  • damping = 0 for perpetual motion, 0.02–0.05 for realistic gradual decay

Environment gravity reference:
  Earth: 9.8 | Moon: 1.6 | Mars: 3.7 | Jupiter: 24.8 | Asteroid: 0.6 | Space: 0.0\
"""

# ---------------------------------------------------------------------------
# 7a — Layout & Composition
# ---------------------------------------------------------------------------

_LAYOUT_SYSTEM = f"""\
You are a 3D scene layout planner — the first stage of a three-stage scene
architect. Your ONLY job is to decide WHAT objects exist and roughly WHERE
they go. Do NOT add materials, fine geometry parameters, animation, lights,
or camera — a later stage handles those.

{_RULES_LAYOUT}

Output ONLY this JSON shape (no markdown, no explanation):
{{
  "objects": [
    {{
      "id": "snake_case_unique",
      "type": "primitive|mesh",
      "geometry": {{"type": "sphere|box|cylinder|ring|torus|capsule|plane"}},
      "model": "/assets/meshes/... (mesh objects only, must be from AVAILABLE MESHES)",
      "position": [x, y, z],
      "parent": "other_id (optional)",
      "label": "Human-readable (optional, for objects a student would ask about)"
    }}
  ]
}}

Rules:
- One entry per distinct object the scene needs, including satellites/rings/
  attached parts as separate entries with "parent" set.
- Meshes: if a concept is in AVAILABLE MESHES, its "type" MUST be "mesh" and
  "model" MUST be that exact path. Never invent a mesh path.
- Objects must NOT all sit at [0,0,0] — spread them in 3D space using the
  scale/orbital-radius reference numbers as a rough guide.
- Every id must be unique snake_case.
"""


def _architect_layout(
    optimized_prompt: str,
    intent_ir: dict[str, Any],
    verified_assets: list[dict[str, str]],
    run_id: str,
    on_event: OnEvent | None,
    feedback: str | None = None,
    previous_layout: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    `feedback`/`previous_layout` are set only when the intent verifier
    (pipeline/intent_verifier.py, Stage 8.5) determined this pass needs to
    be re-run to better satisfy the user's original request — the feedback
    text is injected into the prompt as a correction instruction, and the
    previous layout is shown so the model can adjust rather than start over
    from nothing.
    """
    emit = make_emitter(run_id, on_event)
    label = "Architect — Layout & Composition" + (" (revision)" if feedback else "")
    emit("architect_layout", STARTED, label)
    t0 = time.monotonic()

    from pipeline.asset_registry import build_mesh_menu
    mesh_menu = build_mesh_menu(verified_assets)
    mesh_header = (
        "AVAILABLE MESHES — YOU MUST use type=\"mesh\" for every concept listed here.\n"
        "Do NOT use a primitive for these concepts. The GLB files are confirmed on disk.\n"
        if verified_assets
        else "AVAILABLE MESHES (none — build everything from primitives):\n"
    )

    feedback_block = ""
    if feedback:
        feedback_block = (
            f"\n\nREVISION REQUIRED — a reviewer compared your previous layout against the "
            f"user's request and found it doesn't fully satisfy it:\n{feedback}\n\n"
            f"PREVIOUS LAYOUT (fix the issues above, keep what already works, keep ids "
            f"where the object is still correct):\n"
            f"{json.dumps(previous_layout, separators=(',', ':')) if previous_layout else '(none)'}\n"
        )

    prompt = (
        f"{mesh_header}{mesh_menu}\n\n"
        f"USER REQUEST: \"{optimized_prompt}\"\n"
        f"SCENE INTENT: {json.dumps(intent_ir, separators=(',', ':'))}"
        f"{feedback_block}\n\n"
        "Generate the object layout JSON now:"
    )

    raw, provider = call_llm(_MODEL_GEMINI_PRO, _MODEL_GROQ, prompt, _LAYOUT_SYSTEM)
    skeleton = _parse_json(raw) if raw else None
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    if not skeleton or not isinstance(skeleton.get("objects"), list) or not skeleton["objects"]:
        logger.warning("Architect layout: no usable object skeleton produced")
        emit("architect_layout", COMPLETED, label,
             elapsed_ms=elapsed_ms, provider=provider)
        return None

    logger.info("Architect layout: %d object(s) via %s", len(skeleton["objects"]), provider or "none")
    emit("architect_layout", OUTPUT, label,
         payload=skeleton, provider=provider)
    emit("architect_layout", COMPLETED, label,
         elapsed_ms=elapsed_ms, provider=provider)
    return skeleton


# ---------------------------------------------------------------------------
# 7b — Per-Object Detail
# ---------------------------------------------------------------------------

_DETAIL_SYSTEM = f"""\
You are a 3D scene detail architect — the second stage of a three-stage
scene architect. You receive an object LAYOUT (ids, types, coarse positions,
parent/child structure) that is already decided. Your job is to fill in full
detail for every object: exact geometry parameters, materials, animation,
and educational label/description text.

DO NOT add, remove, or wholesale reposition objects — the object list from
LAYOUT is fixed. You MAY make small position/scale adjustments for
correctness (avoiding overlaps, fixing implausible scale) but you must keep
every object id from the input and must not invent new ones.

{_SCHEMA}

{_EDUCATION}

{_PHYSICS}

{_EXAMPLE_ORBITAL}

{_EXAMPLE_ANATOMICAL}

OUTPUT RULES:
- Output ONLY the raw JSON object: {{"objects": [...]}} — same ids as the input layout.
- Every object needs "position", "material", "animation".
- material.color must be a 6-digit hex "#rrggbb". Use diverse, semantically
  correct colors — never all #888888 or #ffffff.
- For type=mesh objects always use color="#ffffff" so embedded GLB textures are not tinted.
- Mesh objects MUST be scaled large enough to fill the view (scale [3,3,3] or
  larger for a single main-subject mesh).
- Every object carrying a "label" MUST also carry a "description" (see EDUCATIONAL OUTPUT).
- NEVER write math expressions or code in JSON values. Pre-compute all numbers yourself.
- Omit any field you are unsure about rather than guessing wrong values.\
"""


def _architect_detail(
    optimized_prompt: str,
    intent_ir: dict[str, Any],
    layout: dict[str, Any],
    run_id: str,
    on_event: OnEvent | None,
    feedback: str | None = None,
) -> dict[str, Any] | None:
    """`feedback` — see _architect_layout()'s docstring; same revision contract."""
    emit = make_emitter(run_id, on_event)
    label = "Architect — Object Detail" + (" (revision)" if feedback else "")
    emit("architect_detail", STARTED, label)
    t0 = time.monotonic()

    feedback_block = ""
    if feedback:
        feedback_block = (
            f"\n\nREVISION REQUIRED — a reviewer compared the scene against the user's "
            f"request and found it doesn't fully satisfy it. Fix these issues while "
            f"filling in detail (still keep the same object ids from LAYOUT):\n{feedback}\n"
        )

    prompt = (
        f"USER REQUEST: \"{optimized_prompt}\"\n"
        f"SCENE INTENT: {json.dumps(intent_ir, separators=(',', ':'))}"
        f"{feedback_block}\n\n"
        f"LAYOUT (fill in full detail for every object below, keep same ids):\n"
        f"{json.dumps(layout, separators=(',', ':'))}\n\n"
        "Generate the fully-detailed objects JSON now:"
    )

    raw, provider = call_llm(_MODEL_GEMINI_PRO, _MODEL_GROQ, prompt, _DETAIL_SYSTEM)
    detailed = _parse_json(raw) if raw else None
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    if not detailed or not isinstance(detailed.get("objects"), list) or not detailed["objects"]:
        logger.warning("Architect detail: no usable detail produced — using layout skeleton as-is")
        emit("architect_detail", COMPLETED, label,
             elapsed_ms=elapsed_ms, provider=provider)
        return None

    logger.info("Architect detail: %d object(s) detailed via %s", len(detailed["objects"]), provider or "none")
    emit("architect_detail", OUTPUT, label,
         payload=detailed, provider=provider)
    emit("architect_detail", COMPLETED, label,
         elapsed_ms=elapsed_ms, provider=provider)
    return detailed


# ---------------------------------------------------------------------------
# 7c — Lighting / Camera / Polish
# ---------------------------------------------------------------------------

_FINISH_SYSTEM = f"""\
You are a 3D scene lighting and camera director — the third and final stage
of a three-stage scene architect. You receive the complete, fully-detailed
object list. Your job is to decide lights, camera framing, and the scene's
title/summary. You do NOT change objects except for minor position/scale
nudges purely for camera-frame fit if truly necessary.

{_RULES_FINISH}

{_EDUCATION}

OUTPUT RULES:
- Output ONLY this JSON: {{"name": "...", "summary": "...", "lights": [...], "camera": {{...}}}}
- "name": 2-5 words, e.g. "Basketball Bouncing on Hardwood".
- "summary": REQUIRED, 2-4 sentences (see EDUCATIONAL OUTPUT above).
- Include at least 3 lights (ambient + key directional + fill/point) for depth and realism.
- Camera must frame the entire scene; never place it inside objects or too close.\
"""


def _architect_finish(
    optimized_prompt: str,
    intent_ir: dict[str, Any],
    objects: list[dict[str, Any]],
    run_id: str,
    on_event: OnEvent | None,
    feedback: str | None = None,
) -> dict[str, Any] | None:
    """`feedback` — see _architect_layout()'s docstring; same revision contract."""
    emit = make_emitter(run_id, on_event)
    label = "Architect — Lighting, Camera & Polish" + (" (revision)" if feedback else "")
    emit("architect_finish", STARTED, label)
    t0 = time.monotonic()

    feedback_block = ""
    if feedback:
        feedback_block = (
            f"\n\nREVISION REQUIRED — a reviewer found the scene doesn't fully satisfy "
            f"the user's request with respect to mood/lighting/camera framing. Fix:\n{feedback}\n"
        )

    prompt = (
        f"USER REQUEST: \"{optimized_prompt}\"\n"
        f"SCENE INTENT: {json.dumps(intent_ir, separators=(',', ':'))}"
        f"{feedback_block}\n\n"
        f"COMPLETE OBJECT LIST:\n{json.dumps(objects, separators=(',', ':'))}\n\n"
        "Generate the name/summary/lights/camera JSON now:"
    )

    raw, provider = call_llm(
        _MODEL_GEMINI_FLASH, _MODEL_GROQ, prompt, _FINISH_SYSTEM, temperature=0.4
    )
    finish = _parse_json(raw) if raw else None
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    if not finish or not isinstance(finish.get("lights"), list) or not finish.get("camera"):
        logger.warning("Architect finish: no usable lighting/camera produced")
        emit("architect_finish", COMPLETED, label,
             elapsed_ms=elapsed_ms, provider=provider)
        return None

    logger.info("Architect finish: '%s' via %s", finish.get("name", "?"), provider or "none")
    emit("architect_finish", OUTPUT, label,
         payload=finish, provider=provider)
    emit("architect_finish", COMPLETED, label,
         elapsed_ms=elapsed_ms, provider=provider)
    return finish


# ---------------------------------------------------------------------------
# Deterministic safety net for 7b/7c partial failures
# ---------------------------------------------------------------------------

def _default_lighting_camera(objects: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Hardcoded safe 3-light rig + bounding-box-derived camera, used when the
    finish pass (7c) fails but we already have valid objects from 7a/7b. No
    LLM call — pure deterministic fallback so a transient lighting-pass
    failure doesn't throw away an otherwise-good scene.
    """
    xs, ys, zs = [], [], []
    for obj in objects:
        pos = obj.get("position")
        if isinstance(pos, list) and len(pos) == 3 and all(isinstance(v, (int, float)) for v in pos):
            xs.append(pos[0])
            ys.append(pos[1])
            zs.append(pos[2])
    cx = sum(xs) / len(xs) if xs else 0.0
    cy = sum(ys) / len(ys) if ys else 0.0
    cz = sum(zs) / len(zs) if zs else 0.0
    radius = max(
        [((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) ** 0.5 for x, y, z in zip(xs, ys, zs)]
        or [5.0]
    )
    radius = max(radius, 3.0)

    return {
        "name": "Generated Scene",
        "summary": "",
        "lights": [
            {"type": "ambient", "intensity": 0.35, "color": "#ffffff"},
            {"type": "directional", "intensity": 1.2, "color": "#ffffff",
             "position": [cx + radius, cy + radius, cz + radius], "castShadow": True},
            {"type": "point", "intensity": 1.0, "color": "#ffffff",
             "position": [cx, cy + radius * 0.5, cz]},
        ],
        "camera": {
            "position": [cx, cy + radius * 0.6, cz + radius * 2.0],
            "target": [cx, cy, cz],
            "fov": 60,
        },
    }


# ---------------------------------------------------------------------------
# Public API — orchestrates 7a -> 7b -> 7c
# ---------------------------------------------------------------------------

def generate_scene(
    transcript: str,
    intent: dict[str, Any],
    verified_assets: list[dict[str, str]],
    intent_ir: dict[str, Any] | None = None,
    run_id: str = "",
    on_event: OnEvent | None = None,
) -> dict[str, Any] | None:
    """
    Build the full scene JSON via three sequential LLM passes (layout ->
    detail -> finish). Returns the assembled scene dict, or None if the
    layout pass (7a) fails to produce any objects — that failure propagates
    to the caller so it can fall back to the legacy retrieval/builder path.

    `intent` is the resolved semantic-parser intent (objects/structures/
    systems/effects concept buckets) — kept for API-compatibility with
    existing call sites and still logged, but the primary source of
    high-level guidance is now `intent_ir` (the Scene Intent IR from
    pipeline/intent_extractor.py).
    """
    intent_ir = intent_ir or {}
    optimized_prompt = transcript

    logger.info("Scene architect: building scene for '%s'", optimized_prompt[:60])

    layout = _architect_layout(optimized_prompt, intent_ir, verified_assets, run_id, on_event)
    if not layout:
        logger.warning("Scene architect: layout pass failed — architect stage aborted")
        return None

    detailed = _architect_detail(optimized_prompt, intent_ir, layout, run_id, on_event)
    objects = detailed["objects"] if detailed else layout["objects"]

    finish = _architect_finish(optimized_prompt, intent_ir, objects, run_id, on_event)
    if not finish:
        finish = _default_lighting_camera(objects)

    scene: dict[str, Any] = {
        "name": finish.get("name", "Generated Scene"),
        "summary": finish.get("summary", ""),
        "objects": objects,
        "lights": finish.get("lights", []),
        "camera": finish.get("camera", {}),
    }

    allowed_paths = {a["path"] for a in verified_assets}
    scene = _sanitize_mesh_paths(scene, allowed_paths)
    scene = _fix_plane_intersections(scene)

    logger.info(
        "Scene architect: built '%s' with %d objects",
        scene.get("name", "?"),
        len(scene.get("objects", [])),
    )
    return scene


# ---------------------------------------------------------------------------
# Public API — targeted re-run of a single pass, driven by intent-verifier
# feedback (pipeline/intent_verifier.py, Stage 8.5). This is what makes the
# feedback loop a genuine "modify and realign" mechanism rather than a
# fallback-to-primitives escape hatch: only the specific pass judged to be
# the source of the mismatch is re-run, with the reviewer's feedback text
# injected into its prompt, and the result is merged back into the existing
# scene rather than starting over.
# ---------------------------------------------------------------------------

def regenerate_pass(
    pass_name: str,
    scene: dict[str, Any],
    optimized_prompt: str,
    intent_ir: dict[str, Any],
    verified_assets: list[dict[str, str]],
    feedback: str,
    run_id: str = "",
    on_event: OnEvent | None = None,
) -> dict[str, Any] | None:
    """
    Re-run exactly one architect pass ("layout" | "detail" | "finish")
    against the current `scene`, with `feedback` (a reviewer's description
    of what's wrong) injected into that pass's prompt, then merge the result
    back into `scene`. Returns the updated scene, or None if the targeted
    pass itself fails to produce anything usable (caller should keep the
    pre-regeneration scene in that case).

    Re-running "layout" cascades: since object identity/positions may change,
    detail and finish are re-run afterward too (without additional feedback,
    just to stay consistent with the new layout) — otherwise a layout change
    could leave stale materials/lighting behind. Re-running "detail" or
    "finish" alone does not cascade, since those passes don't change object
    identity.
    """
    objects = scene.get("objects", [])

    if pass_name == "layout":
        previous_layout = {"objects": objects}
        new_layout = _architect_layout(
            optimized_prompt, intent_ir, verified_assets, run_id, on_event,
            feedback=feedback, previous_layout=previous_layout,
        )
        if not new_layout:
            logger.warning("regenerate_pass(layout): produced nothing usable — keeping prior scene")
            return None
        detailed = _architect_detail(optimized_prompt, intent_ir, new_layout, run_id, on_event)
        new_objects = detailed["objects"] if detailed else new_layout["objects"]
        finish = _architect_finish(optimized_prompt, intent_ir, new_objects, run_id, on_event)
        if not finish:
            finish = _default_lighting_camera(new_objects)

    elif pass_name == "detail":
        layout = {"objects": objects}
        detailed = _architect_detail(
            optimized_prompt, intent_ir, layout, run_id, on_event, feedback=feedback,
        )
        if not detailed:
            logger.warning("regenerate_pass(detail): produced nothing usable — keeping prior scene")
            return None
        new_objects = detailed["objects"]
        finish = {
            "name": scene.get("name", "Generated Scene"),
            "summary": scene.get("summary", ""),
            "lights": scene.get("lights", []),
            "camera": scene.get("camera", {}),
        }

    elif pass_name == "finish":
        new_objects = objects
        finish = _architect_finish(
            optimized_prompt, intent_ir, objects, run_id, on_event, feedback=feedback,
        )
        if not finish:
            logger.warning("regenerate_pass(finish): produced nothing usable — keeping prior scene")
            return None

    else:
        logger.warning("regenerate_pass: unknown pass_name %r", pass_name)
        return None

    updated: dict[str, Any] = {
        "name": finish.get("name", scene.get("name", "Generated Scene")),
        "summary": finish.get("summary", scene.get("summary", "")),
        "objects": new_objects,
        "lights": finish.get("lights", scene.get("lights", [])),
        "camera": finish.get("camera", scene.get("camera", {})),
    }

    allowed_paths = {a["path"] for a in verified_assets}
    updated = _sanitize_mesh_paths(updated, allowed_paths)
    updated = _fix_plane_intersections(updated)

    logger.info(
        "regenerate_pass(%s): scene updated, '%s' with %d objects",
        pass_name, updated.get("name", "?"), len(updated.get("objects", [])),
    )
    return updated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fix_plane_intersections(scene: dict[str, Any]) -> dict[str, Any]:
    """
    Move horizontal planes that intersect non-plane objects downward so they
    don't slice through meshes/spheres. Runs after the finish pass, before
    validation.
    """
    objects = scene.get("objects", [])
    planes = [o for o in objects if isinstance(o, dict)
              and o.get("geometry", {}).get("type") == "plane"]
    non_planes = [o for o in objects if isinstance(o, dict)
                  and o.get("geometry", {}).get("type") != "plane"
                  and o.get("type") != "plane"]

    if not planes or not non_planes:
        return scene

    # Find the Y extent of all non-plane objects
    min_y = min(
        (o.get("position") or [0, 0, 0])[1]
        for o in non_planes
        if isinstance(o.get("position"), list) and len(o["position"]) >= 2
    )

    for plane in planes:
        pos = plane.get("position")
        if not isinstance(pos, list) or len(pos) < 3:
            continue
        plane_y = pos[1]
        # If plane y is within 1 unit of the lowest object, push it below
        if plane_y > min_y - 1.0:
            new_y = min_y - 2.0
            plane["position"] = [pos[0], new_y, pos[2]]
            logger.debug(
                "Moved plane '%s' from y=%.2f to y=%.2f to avoid intersection",
                plane.get("id"), plane_y, new_y,
            )

    return scene


def _clean_json(text: str) -> str:
    """Strip common LLM JSON artifacts: markdown fences, trailing commas, // comments."""
    # Strip markdown fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            stripped = part.strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            if stripped.startswith("{"):
                text = stripped
                break
    # Remove // line comments (outside strings — naive but good enough)
    text = re.sub(r'//[^\n"]*\n', '\n', text)
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([\}\]])', r'\1', text)
    return text


def _parse_json(raw: str) -> dict[str, Any] | None:
    try:
        text = raw.strip()
        text = _clean_json(text)
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end <= start:
            return None
        return json.loads(text[start:end])
    except Exception as exc:
        logger.debug("JSON parse error: %s", exc)
        return None


def _is_numeric_xyz(val: Any) -> bool:
    """True if val is a list of exactly 3 finite numbers."""
    return (
        isinstance(val, list)
        and len(val) == 3
        and all(isinstance(v, (int, float)) and v == v for v in val)  # v==v rejects NaN
    )


def _normalize_object(obj: dict[str, Any]) -> dict[str, Any]:
    """Fix common LLM output issues before validation."""
    # scale as a scalar → [n, n, n]
    sc = obj.get("scale")
    if isinstance(sc, (int, float)):
        obj["scale"] = [sc, sc, sc]
    # position as a scalar or invalid → [0, 0, 0]
    pos = obj.get("position")
    if not _is_numeric_xyz(pos):
        obj["position"] = [0.0, 0.0, 0.0]
    # animation missing → none
    if "animation" not in obj:
        obj["animation"] = {"type": "none"}
    # material.color must be hex — sometimes the LLM outputs "red"
    mat = obj.get("material")
    if isinstance(mat, dict):
        color = mat.get("color", "")
        if not (isinstance(color, str) and color.startswith("#") and len(color) == 7):
            # Mesh objects keep #ffffff so embedded textures show through
            mat["color"] = "#ffffff" if obj.get("type") == "mesh" else "#888888"
    # geometry: fix cylinder from/to if they contain non-numeric elements
    geom = obj.get("geometry")
    if isinstance(geom, dict) and geom.get("type") == "cylinder":
        for field in ("from", "to"):
            val = geom.get(field)
            if not _is_numeric_xyz(val):
                logger.debug(
                    "Dropping object '%s': cylinder.%s has non-numeric value %r",
                    obj.get("id"), field, val,
                )
                obj["_drop"] = True
                return obj
    return obj


def _sanitize_mesh_paths(scene: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    """
    Convert any object with an unrecognised mesh path to a fallback primitive
    (grey sphere) rather than letting the validator reject the whole scene.
    """
    sanitized = []
    for obj in scene.get("objects", []):
        obj = _normalize_object(obj)
        if not isinstance(obj, dict):
            continue
        if obj.pop("_drop", False):
            continue
        if obj.get("type") == "mesh":
            path = obj.get("model", "")
            if path not in allowed:
                logger.debug(
                    "Replacing unverified mesh '%s' with primitive fallback", path
                )
                obj = {
                    "id":       obj.get("id", "fallback"),
                    "type":     "primitive",
                    "geometry": {"type": "sphere", "radius": 1.0},
                    "position": obj.get("position", [0, 0, 0]),
                    "scale":    obj.get("scale", [1, 1, 1]),
                    "material": obj.get("material") or {
                        "type": "standard", "color": "#888888",
                        "roughness": 0.6, "metalness": 0.1,
                    },
                    "animation": obj.get("animation") or {"type": "none"},
                    **({"label":  obj["label"]}  if obj.get("label")  else {}),
                    **({"description": obj["description"]} if obj.get("description") else {}),
                    **({"parent": obj["parent"]} if obj.get("parent") else {}),
                }
        sanitized.append(obj)
    scene["objects"] = sanitized
    return scene
