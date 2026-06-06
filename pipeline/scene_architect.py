"""
Scene architect: one Groq call that produces the entire scene JSON.

Groq decides:
  - Which objects use primitives vs verified GLB meshes
  - Realistic sizes, colors, and relative proportions
  - Correct lighting for the scene type
  - Camera position that frames everything properly
  - Parent-child relationships (moons, rings, attached parts)
  - Animations (orbits with correct relative speeds, spins)
"""
from __future__ import annotations

import json
import os
import requests
from typing import Any

from core.utils.logger import get_logger

logger = get_logger("scene_architect")

# ── Model config ──────────────────────────────────────────────────────────────
# Primary: Gemini 2.5 Pro via direct API (GEMINI_API_KEY)
# Fallback: Groq Llama 3.3 70B (GROQ_API_KEY)
_GROQ_MODEL = "llama-3.3-70b-versatile"
_GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"

try:
    from google import genai as _genai
    from google.genai.types import HttpOptions as _HttpOptions
    from google.genai import types as _genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


def _make_vertex_client():
    """Return a Vertex AI genai client using ADC (no API key required)."""
    return _genai.Client(
        vertexai=True,
        project=os.getenv("GCP_PROJECT", "reportevaluator"),
        location=os.getenv("GCP_LOCATION", "us-central1"),
        http_options=_HttpOptions(api_version="v1"),
    )


def _call_architect_gemini(prompt: str, system: str) -> str | None:
    """Gemini 2.5 Pro via Vertex AI (ADC). Returns raw text or None."""
    if not _GENAI_AVAILABLE:
        return None
    model = os.getenv("GEMINI_ARCHITECT_MODEL", "gemini-2.5-pro")
    try:
        client = _make_vertex_client()
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=_genai_types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                temperature=0.4,
            ),
        )
        return response.text
    except Exception as exc:
        logger.error("Architect Gemini call failed: %s", exc)
        return None


def _call_architect_groq(prompt: str, system: str) -> str | None:
    """Groq Llama 3.3 70B fallback with json_object response format enforced."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": _GROQ_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.4,
    }
    try:
        resp = requests.post(_GROQ_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.error("Architect Groq call failed: %s", exc)
        return None


def _call_architect(prompt: str, system: str) -> str | None:
    """Try Gemini 2.5 Pro first; fall back to Groq on any failure."""
    raw = _call_architect_gemini(prompt, system)
    if raw:
        logger.info("Scene architect: using Gemini")
        return raw
    logger.info("Scene architect: Gemini unavailable, falling back to Groq")
    return _call_architect_groq(prompt, system)

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
    "label": "Human-readable"   ← optional, show for key objects only
  }

Object (mesh — ONLY if path appears in AVAILABLE MESHES below):
  { "id": "...", "type": "mesh", "model": "/assets/meshes/...",
    "position": [...], "scale": [...],
    "material": { "type":"standard", "color":"#ffffff", "roughness":0.5, "metalness":0.0 },
    "animation": {...}, "label": "..." }
  IMPORTANT: For mesh objects always use color "#ffffff" — the GLB has embedded textures;
  any other color will tint/destroy the original appearance.

Lights:
  { "type":"ambient",     "intensity":0-2,  "color":"#rrggbb" }
  { "type":"directional", "intensity":0-3,  "color":"#rrggbb", "position":[x,y,z], "castShadow":true }
  { "type":"point",       "intensity":0-10, "color":"#rrggbb", "position":[x,y,z] }

Camera: { "position":[x,y,z], "target":[x,y,z], "fov":40-75 }\
"""

# ---------------------------------------------------------------------------
# Decision, lighting, and scale rules
# ---------------------------------------------------------------------------

_RULES = """\
═══ PRIMITIVE vs MESH ═══
Use primitives for:
  • ALL astronomical: stars, planets, moons, rings, asteroid belts, galaxies, nebulae
  • ALL atomic / molecular: nucleus, protons, neutrons, electrons, bonds, shells
  • ALL biological organs built from compound parts: heart chambers, blood vessels,
    brain lobes, lungs, kidneys, neurons, DNA helix
  • ALL mechanical/structural: gears, orreries, clock faces, crystal lattices,
    bridges, towers, molecules, grids
  • ANY abstract concept, energy field, wave, or geometric arrangement

Use a MESH only when:
  • The exact concept name appears in AVAILABLE MESHES below
  • A sphere/box would lose the essential recognisable silhouette (dragon, spaceship,
    human figure, complex creature)
  • NEVER invent a mesh path — only use paths listed in AVAILABLE MESHES

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

═══ SCALE AND CAMERA ═══
Objects MUST be sized relative to each other — never all the same size.

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
  Bond1: cylinder from=[0,0,0] to=[0.754,-0.588,0] r=0.08
  Bond2: cylinder from=[0,0,0] to=[-0.754,-0.588,0] r=0.08

CRITICAL — PRE-COMPUTED NUMBERS ONLY:
  ALL numeric values (positions, radii, from/to, scales) MUST be pre-computed decimal numbers.
  NEVER write math expressions, formulas, trig calls, or code of any kind.
  RIGHT: "to": [0.754, -0.588, 0]
  WRONG: "to": [0.96 * Math.sin(52 * Math.PI / 180), ...]
  WRONG: "to": ["sin(52°) * 0.96", ...]
  Compute the result yourself before writing. Write only the final number.

Camera positioning:
  • Find the bounding box of all objects
  • scene_radius = max(all object distances from center)
  • camera.position = [cx, cy + scene_radius*0.6, cz + scene_radius*2.0]
  • camera.target = scene center
  • fov: 60-65 for orbital scenes, 45-55 for anatomical close-up, 55-65 general
  • Never place camera inside the scene or behind objects

Parent-child usage:
  • Moon → parent: "earth"  (position relative to earth)
  • Saturn ring → parent: "saturn"
  • Electron → parent: none (use orbit animation with center=[0,0,0] for shells)
  • Attached mechanical arms → parent: "base"\
"""

# ---------------------------------------------------------------------------
# Inline quality examples (few-shot)
# ---------------------------------------------------------------------------

_EXAMPLE_ORBITAL = """\
EXAMPLE — "solar system" (orbital primitive scene, use as quality reference):
{
  "name": "Solar System",
  "camera": {"position":[0,18,55], "target":[0,0,0], "fov":60},
  "lights": [
    {"type":"ambient","intensity":0.12,"color":"#ffffff"},
    {"type":"point","intensity":5.0,"color":"#ffcc66","position":[0,0,0]},
    {"type":"directional","intensity":0.5,"color":"#ffffff","position":[50,30,20],"castShadow":true}
  ],
  "objects": [
    {"id":"sun","type":"primitive","geometry":{"type":"sphere","radius":2.5},"position":[0,0,0],"scale":[1,1,1],"material":{"type":"standard","color":"#ffdd33","roughness":0.4,"metalness":0.0,"emissive":"#ff8800","emissiveIntensity":1.0},"label":"Sun","animation":{"type":"none"}},
    {"id":"earth","type":"primitive","geometry":{"type":"sphere","radius":0.5},"position":[9,0,0],"scale":[1,1,1],"material":{"type":"standard","color":"#1a5acc","roughness":0.7,"metalness":0.05},"label":"Earth","animation":{"type":"orbit","center":[0,0,0],"speed":1.0,"phase":0}},
    {"id":"moon","type":"primitive","geometry":{"type":"sphere","radius":0.14},"position":[0.65,0,0],"scale":[1,1,1],"parent":"earth","material":{"type":"standard","color":"#cccccc","roughness":0.95,"metalness":0.0},"animation":{"type":"orbit","center":[0,0,0],"speed":13.0}},
    {"id":"saturn","type":"primitive","geometry":{"type":"sphere","radius":1.0},"position":[21,0,0],"scale":[1,1,1],"material":{"type":"standard","color":"#ddcc88","roughness":0.5,"metalness":0.05},"label":"Saturn","animation":{"type":"orbit","center":[0,0,0],"speed":0.323,"phase":1.2}},
    {"id":"saturn_ring","type":"primitive","geometry":{"type":"ring","innerRadius":1.35,"outerRadius":2.35,"thetaSegments":128},"position":[0,0,0],"rotation":[16,0,8],"scale":[1,1,1],"parent":"saturn","material":{"type":"standard","color":"#ccbb77","roughness":0.8,"metalness":0.0,"opacity":0.6,"transparent":true},"animation":{"type":"none"}}
  ]
}
END EXAMPLE\
"""

_EXAMPLE_ANATOMICAL = """\
EXAMPLE — "human heart" (anatomical primitive scene, compound shapes):
Key objects (abbreviated — full scene uses ~25 primitives):
  left_ventricle_core: sphere r=1.55, scale=[0.95,1.34,1.06], color=#8f1a1a, pos=[-0.62,-1.35,0.12]
  right_ventricle_core: sphere r=1.24, color=#6f1014
  aorta_root: cylinder from=[-0.35,1.02,0.08] to=[-0.35,2.22,0.1] r=0.38, color=#c23434
  pericardium: sphere r=2.48, opacity=0.18, transparent=true (outer membrane)
  Camera: position=[0.2,1.4,11.8], target=[0,-0.4,0.2], fov=46
  Lights: warm ambient 0.38 + 2 directionals (key + fill) + subtle blue point
END EXAMPLE\
"""

# ---------------------------------------------------------------------------
# Physics animation prompt block
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
  • floor_y MUST always be strictly below the object's starting y position
    Example: object at y=2 → floor_y must be < 2, e.g. -1 or 0
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
# System prompt assembly
# ---------------------------------------------------------------------------

_SYSTEM = f"""\
You are a 3D holographic scene architect. Given a user request, output ONE complete scene JSON.
Your output is rendered directly in Three.js — correctness and visual quality matter.

{_RULES}

{_SCHEMA}

{_PHYSICS}

{_EXAMPLE_ORBITAL}

{_EXAMPLE_ANATOMICAL}

OUTPUT RULES:
- Output ONLY the raw JSON object. No markdown fences, no explanation.
- Every object needs a unique "id" in snake_case.
- Every object needs "position", "material", "animation".
- material.color must be a 6-digit hex "#rrggbb".
- Use diverse, semantically correct colors — never all #888888 or #ffffff.
- For type=mesh objects always use color="#ffffff" so embedded GLB textures are not tinted.
- Include at least 3 lights (ambient + key directional + fill/point) for depth and realism.
- Objects must NOT all sit at [0,0,0] — spread them naturally in 3D space.
- Mesh objects should be sized visually large enough to be seen (scale ≥ [1,1,1] unless tiny).
- Camera must frame the entire scene; never place it inside objects or too close.
- If you add a ground/water/floor plane, it MUST be positioned so it does NOT intersect
  any other object. Place it clearly below (or above) all objects with vertical clearance.
  Example: mermaid at y=0, water surface plane at y=-2 (below feet) or y=4 (above head).
  NEVER place a horizontal plane at the same y as an object's center — it will slice it.
- Omit any field you are unsure about rather than guessing wrong values.
- NEVER write math expressions or code in JSON values. Pre-compute all numbers yourself.\
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_scene(
    transcript: str,
    intent: dict[str, Any],
    verified_assets: list[dict[str, str]],
) -> dict[str, Any] | None:
    """
    Ask Groq to build the full scene from scratch.
    Returns parsed dict or None on failure.
    """
    key = os.getenv("GROQ_API_KEY")
    if not key:
        logger.warning("GROQ_API_KEY not set — scene architect skipped")
        return None

    from pipeline.asset_registry import build_mesh_menu
    mesh_menu = build_mesh_menu(verified_assets)

    intent_summary = {
        k: v for k, v in intent.items()
        if v and not k.startswith("_")
    }

    prompt = (
        f"AVAILABLE MESHES (use ONLY these paths for type=mesh, nothing else):\n"
        f"{mesh_menu}\n\n"
        f"USER REQUEST: \"{transcript}\"\n"
        f"SEMANTIC INTENT: {json.dumps(intent_summary)}\n\n"
        "Generate the complete scene JSON now:"
    )

    logger.info("Scene architect: building scene for '%s'", transcript[:60])
    raw = _call_architect(prompt, _SYSTEM)
    if not raw:
        logger.warning("Scene architect: Groq returned nothing")
        return None

    scene = _parse_json(raw)
    if not scene:
        logger.warning("Scene architect: could not parse Groq JSON response")
        return None

    # Safety: strip any mesh paths that are not in the verified list
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
# Helpers
# ---------------------------------------------------------------------------

def _fix_plane_intersections(scene: dict[str, Any]) -> dict[str, Any]:
    """
    Move horizontal planes that intersect non-plane objects downward so they
    don't slice through meshes/spheres. Runs after Groq output, before validation.
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
    import re
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
        end   = text.rfind("}") + 1
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
    # material.color must be hex — sometimes Groq outputs "red"
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
                    **({"parent": obj["parent"]} if obj.get("parent") else {}),
                }
        sanitized.append(obj)
    scene["objects"] = sanitized
    return scene
