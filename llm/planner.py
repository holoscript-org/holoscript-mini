"""llm/planner.py
Stage 1 of the Member 1 pipeline: Extract abstract scene plan from user command.

The planner does NOT generate final JSON. Instead, it produces a structured
plan that describes what the scene should contain:
  - scene_type (atom, molecule, solar_system, etc.)
  - num_objects (estimated count)
  - components (list of what to build)
  - animation_types (which objects should move)
  - hierarchy_needed (do we need parent-child relationships?)
  - use_mesh (should any part use a mesh model instead of primitives?)
  - complexity (low/medium/high)

The parametric generator then uses this plan to compute exact coordinates.
The builder then uses both plan + coordinates to generate final objects.
"""

import json
import time
import os
import hashlib
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator
from core.utils.logger import get_logger

load_dotenv()

logger = get_logger("planner")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"


# ---------------------------------------------------------------------------
# Planner Output Schema
# ---------------------------------------------------------------------------

class ScenePlan(BaseModel):
    """Abstract scene plan extracted from user command."""

    scene_type: str
    # Examples: "atom", "molecule", "solar_system", "mechanical", "organic", "abstract"

    description: str
    # Plain language summary of what the user wants

    num_objects: int
    # Estimated total object count (including animations, decorative elements)

    components: list[str]
    # List of major components. Use unique role names, not repeated items.
    # These become the basis for deterministic naming and parametric generation.

    repeat_counts: dict[str, int] = Field(default_factory=dict)
    # Optional counts for repeated components. Example: {"planet": 8, "moon": 1}

    animation_types: list[str]
    # Which animation types appear. Options: "none", "orbit", "spin"

    hierarchy_needed: bool
    # True if parent-child relationships are needed (e.g., moon orbits earth)

    layout_strategy: str = "generic"
    # High-level spatial arrangement: orbit, helix, grid, cluster, ring, spine, scatter, branching

    camera_intent: str = "balanced"
    # Camera framing: close, balanced, wide, cinematic, top_down

    lighting_style: str = "neutral"
    # Lighting mood: neutral, warm, cool, dramatic, neon, clinical

    style_hints: list[str] = Field(default_factory=list)
    # Optional style descriptors that help the deterministic builder choose materials

    color_palette: list[str] = Field(default_factory=list)
    # Optional hex colors chosen by the planner for deterministic material assignment

    component_colors: dict[str, str] = Field(default_factory=dict)
    # Optional mapping of component roles to specific hex colors (e.g. {"earth": "#4488ff", "mars": "#ff6644"})

    component_sizes: dict[str, float] = Field(default_factory=dict)
    # Optional mapping of component roles to size scale multipliers (e.g. {"sun": 2.0, "mercury": 0.3, "jupiter": 3.5})

    component_parent: dict[str, str] = Field(default_factory=dict)
    # Optional mapping of component roles to parent roles for hierarchy (e.g. {"saturn_rings": "saturn", "earth_moon": "earth"})

    focal_object: str | None = None
    # Optional role to center the composition around

    use_mesh: bool
    # True if any component should use a mesh model (organic shapes, complex geometry)

    complexity: str
    # "low" (1-5 objects), "medium" (6-15), "high" (16-20)

    @field_validator("scene_type")
    @classmethod
    def validate_scene_type(cls, v: str) -> str:
        valid = {
            "atom", "molecule", "solar_system", "mechanical", "organic",
            "abstract", "geometric", "crystalline", "astronomical",
            "system", "structure", "vehicle", "character", "landscape", "diagram"
        }
        if v not in valid:
            raise ValueError(f"scene_type must be one of {valid}, got {v}")
        return v

    @field_validator("animation_types")
    @classmethod
    def validate_animation_types(cls, v: list[str]) -> list[str]:
        valid = {"none", "orbit", "spin"}
        for anim_type in v:
            if anim_type not in valid:
                raise ValueError(f"animation_type must be one of {valid}, got {anim_type}")
        return v

    @field_validator("complexity")
    @classmethod
    def validate_complexity(cls, v: str) -> str:
        valid = {"low", "medium", "high"}
        if v not in valid:
            raise ValueError(f"complexity must be one of {valid}, got {v}")
        return v

    @field_validator("layout_strategy")
    @classmethod
    def validate_layout_strategy(cls, v: str) -> str:
        valid = {"generic", "orbit", "helix", "grid", "cluster", "ring", "spine", "scatter", "branching"}
        value = v.strip().lower()
        if value not in valid:
            return "generic"
        return value

    @field_validator("camera_intent")
    @classmethod
    def validate_camera_intent(cls, v: str) -> str:
        valid = {"close", "balanced", "wide", "cinematic", "top_down"}
        value = v.strip().lower()
        if value not in valid:
            return "balanced"
        return value

    @field_validator("lighting_style")
    @classmethod
    def validate_lighting_style(cls, v: str) -> str:
        valid = {"neutral", "warm", "cool", "dramatic", "neon", "clinical"}
        value = v.strip().lower()
        if value not in valid:
            return "neutral"
        return value

    @field_validator("style_hints")
    @classmethod
    def validate_style_hints(cls, v: list[str]) -> list[str]:
        cleaned: list[str] = []
        for hint in v:
            text = str(hint).strip().lower()
            if text and text not in cleaned:
                cleaned.append(text)
        return cleaned

    @field_validator("color_palette")
    @classmethod
    def validate_color_palette(cls, v: list[str]) -> list[str]:
        cleaned: list[str] = []
        for color in v:
            text = str(color).strip().lower()
            if not text:
                continue
            if not (text.startswith("#") and len(text) == 7):
                continue
            if text not in cleaned:
                cleaned.append(text)
        return cleaned

    @field_validator("component_colors")
    @classmethod
    def validate_component_colors(cls, v: dict[str, str]) -> dict[str, str]:
        """Ensure all component colors are valid hex codes."""
        cleaned: dict[str, str] = {}
        for role, color in v.items():
            role_text = str(role).strip()
            color_text = str(color).strip().lower()
            if not role_text or not color_text:
                continue
            if not (color_text.startswith("#") and len(color_text) == 7):
                continue
            cleaned[role_text] = color_text
        return cleaned

    @field_validator("component_sizes")
    @classmethod
    def validate_component_sizes(cls, v: dict[str, float]) -> dict[str, float]:
        """Ensure all component sizes are positive floats."""
        cleaned: dict[str, float] = {}
        for role, size in v.items():
            role_text = str(role).strip()
            if not role_text:
                continue
            try:
                size_float = float(size)
                if size_float > 0:
                    cleaned[role_text] = round(size_float, 2)
            except (ValueError, TypeError):
                continue
        return cleaned

    @field_validator("component_parent")
    @classmethod
    def validate_component_parent(cls, v: dict[str, str]) -> dict[str, str]:
        """Ensure all parent relationships map strings to strings."""
        cleaned: dict[str, str] = {}
        for child, parent in v.items():
            child_text = str(child).strip()
            parent_text = str(parent).strip()
            if child_text and parent_text and child_text != parent_text:
                cleaned[child_text] = parent_text
        return cleaned

    @field_validator("focal_object")
    @classmethod
    def validate_focal_object(cls, v: str | None) -> str | None:
        if v is None:
            return None
        text = str(v).strip()
        return text or None

    @field_validator("num_objects")
    @classmethod
    def validate_num_objects(cls, v: int) -> int:
        if not (1 <= v <= 20):
            raise ValueError(f"num_objects must be between 1 and 20, got {v}")
        return v

    @field_validator("repeat_counts")
    @classmethod
    def validate_repeat_counts(cls, v: dict[str, int]) -> dict[str, int]:
        for key, value in v.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("repeat_counts keys must be non-empty strings")
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"repeat_counts[{key!r}] must be a non-negative integer")
        return v

    @model_validator(mode="after")
    def normalize_components(self) -> "ScenePlan":
        if not self.components:
            raise ValueError("components must not be empty")

        counts: dict[str, int] = dict(self.repeat_counts)
        normalized: list[str] = []

        for component in self.components:
            if component not in normalized:
                normalized.append(component)
            else:
                counts[component] = counts.get(component, 1) + 1

        self.components = normalized
        self.repeat_counts = counts
        return self


VALID_SCENE_TYPES = {
    "atom", "molecule", "solar_system", "mechanical", "organic",
    "abstract", "geometric", "crystalline", "astronomical",
    "system", "structure", "vehicle", "character", "landscape", "diagram",
}

VALID_ANIMATION_TYPES = {"none", "orbit", "spin"}


def _stable_int(*parts: str) -> int:
    data = "|".join(parts).encode("utf-8")
    return int(hashlib.sha1(data).hexdigest()[:8], 16)


def _hex_palette(seed_text: str, count: int = 4) -> list[str]:
    seed = _stable_int(seed_text)
    palette: list[str] = []
    for index in range(count):
        value = (seed >> (index * 5)) & 0xFFFFFF
        red = 64 + ((value >> 16) & 0xFF) % 160
        green = 64 + ((value >> 8) & 0xFF) % 160
        blue = 64 + (value & 0xFF) % 160
        palette.append(f"#{red:02x}{green:02x}{blue:02x}")
    return palette


def _repair_plan_data(plan_data: dict, command: str) -> dict:
    """Normalize loose model output before validation."""
    repaired = dict(plan_data)

    scene_type = str(repaired.get("scene_type", "abstract")).strip().lower()
    if scene_type not in VALID_SCENE_TYPES:
        scene_type = "abstract"
    repaired["scene_type"] = scene_type

    components = repaired.get("components", [])
    if not isinstance(components, list):
        components = []
    unique_components: list[str] = []
    seen_components: set[str] = set()
    for component in components:
        text = str(component).strip()
        if not text or text in seen_components:
            continue
        seen_components.add(text)
        unique_components.append(text)
    repaired["components"] = unique_components

    repeat_counts = repaired.get("repeat_counts", {})
    if not isinstance(repeat_counts, dict):
        repeat_counts = {}
    cleaned_counts: dict[str, int] = {}
    for key, value in repeat_counts.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count < 0:
            continue
        cleaned_counts[key_text] = count
    repaired["repeat_counts"] = cleaned_counts

    animation_types = repaired.get("animation_types", [])
    if not isinstance(animation_types, list):
        animation_types = []
    cleaned_animations: list[str] = []
    for animation in animation_types:
        anim = str(animation).strip().lower()
        if anim in VALID_ANIMATION_TYPES and anim not in cleaned_animations:
            cleaned_animations.append(anim)
    if not cleaned_animations:
        cleaned_animations = ["none"]
    repaired["animation_types"] = cleaned_animations

    if "use_mesh" in repaired:
        repaired["use_mesh"] = bool(repaired["use_mesh"])
    else:
        repaired["use_mesh"] = scene_type == "organic" and len(repaired.get("components", [])) == 1

    lowered = command.lower()
    if "layout_strategy" not in repaired:
        layout_options = ["generic", "orbit", "helix", "grid", "cluster", "ring", "spine", "scatter", "branching"]
        repaired["layout_strategy"] = layout_options[_stable_int(command, scene_type) % len(layout_options)]

    if "camera_intent" not in repaired:
        camera_options = ["close", "balanced", "wide", "cinematic", "top_down"]
        repaired["camera_intent"] = camera_options[_stable_int(command, scene_type, "camera") % len(camera_options)]

    if "lighting_style" not in repaired:
        lighting_options = ["neutral", "warm", "cool", "dramatic", "neon", "clinical"]
        repaired["lighting_style"] = lighting_options[_stable_int(command, scene_type, "lighting") % len(lighting_options)]

    if "style_hints" not in repaired:
        tokens = [token.strip(".,:;!?()[]{}\"'") for token in lowered.split()]
        hints = [token for token in tokens if len(token) > 4 and token.isalpha()]
        repaired["style_hints"] = list(dict.fromkeys(hints[:4]))

    if "color_palette" not in repaired:
        repaired["color_palette"] = _hex_palette(command + scene_type, 4)

    if "focal_object" not in repaired:
        components = repaired.get("components", [])
        repaired["focal_object"] = components[0] if components else None

    try:
        repaired["num_objects"] = int(repaired.get("num_objects", 1))
    except (TypeError, ValueError):
        repaired["num_objects"] = 1
    repaired["num_objects"] = max(1, min(20, repaired["num_objects"]))

    return repaired


# ---------------------------------------------------------------------------
# Planner Implementation
# ---------------------------------------------------------------------------

def _extract_json_from_text(text: str) -> dict | None:
    """Extract JSON from LLM response (may contain markdown or explanation)."""
    try:
        start_idx = text.find("{")
        if start_idx == -1:
            return None
        end_idx = text.rfind("}") + 1
        if end_idx <= start_idx:
            return None
        json_str = text[start_idx:end_idx]
        return json.loads(json_str)
    except Exception:
        return None


def _call_groq(prompt: str) -> str:
    """Call Groq API with the given prompt (or fallback to other providers)."""
    # Try unified provider first; fall back to direct Groq call if needed
    try:
        from llm.unified_provider import get_unified_provider
        provider = get_unified_provider()
        return provider.call(prompt)
    except Exception:
        # Fallback to direct Groq call for backward compatibility
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not found in environment.")

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a scene planner. Output ONLY valid JSON. No markdown. No explanations.",
                },
                {"role": "user", "content": prompt},
            ],
        }

        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]


def plan(command: str) -> ScenePlan | None:
    """
    Extract an abstract scene plan from a user command.

    Args:
        command: User request (e.g., "show a hydrogen atom")

    Returns:
        ScenePlan object if successful, None if planning fails

    Raises:
        RuntimeError: If GROQ_API_KEY is missing
        requests.exceptions.RequestException: If API call fails
    """
    from llm.prompt_templates import build_planner_prompt

    # If no GROQ_API_KEY is configured, use a local deterministic fallback
    # so tests and offline development can proceed.
    if not GROQ_API_KEY:
        logger.info("planner: GROQ_API_KEY not found, using local fallback planner")
        try:
            from llm.planner import _local_plan  # type: ignore
        except Exception:
            # define inline fallback
            def _local_plan(cmd: str) -> dict:
                c = cmd.lower()
                if "hydrogen" in c or "atom" in c:
                    return {
                        "scene_type": "atom",
                        "description": "A hydrogen atom with a central proton and an orbiting electron",
                        "num_objects": 3,
                        "components": ["proton", "electron", "orbital_ring"],
                        "repeat_counts": {},
                        "animation_types": ["none", "orbit"],
                        "hierarchy_needed": False,
                        "use_mesh": False,
                        "complexity": "low",
                    }
                if "solar" in c or "planet" in c or "sun" in c:
                    return {
                        "scene_type": "solar_system",
                        "description": "A solar system with the sun and orbiting planets",
                        "num_objects": 9,
                        "components": ["sun", "planet", "moon"],
                        "repeat_counts": {"planet": 8},
                        "animation_types": ["none", "orbit"],
                        "hierarchy_needed": True,
                        "use_mesh": False,
                        "complexity": "medium",
                    }
                if "dna" in c or "helix" in c:
                    return {
                        "scene_type": "organic",
                        "description": "A DNA double helix with intertwined strands",
                        "num_objects": 12,
                        "components": ["strand_left", "strand_right", "base_pair"],
                        "repeat_counts": {"base_pair": 10},
                        "animation_types": ["none"],
                        "hierarchy_needed": False,
                        "use_mesh": False,
                        "complexity": "medium",
                    }
                if "heart" in c:
                    return {
                        "scene_type": "organic",
                        "description": "A realistic human heart mesh",
                        "num_objects": 1,
                        "components": ["heart_mesh"],
                        "repeat_counts": {},
                        "animation_types": ["none"],
                        "hierarchy_needed": False,
                        "use_mesh": True,
                        "complexity": "high",
                    }
                # default conservative plan
                return {
                    "scene_type": "abstract",
                    "description": f"Abstract scene for: {command}",
                    "num_objects": 3,
                    "components": ["core", "accent", "ring"],
                    "repeat_counts": {},
                    "animation_types": ["none"],
                    "hierarchy_needed": False,
                    "use_mesh": False,
                    "complexity": "low",
                }

        try:
            plan_dict = _local_plan(command)  # type: ignore
            plan_obj = ScenePlan.model_validate(plan_dict)
            logger.info("planner: local plan scene_type=%s num_objects=%d", plan_obj.scene_type, plan_obj.num_objects)
            return plan_obj
        except Exception as e:
            logger.error("planner: local fallback failed: %s", e)
            return None

    prompt = build_planner_prompt(command)
    start = time.perf_counter()

    try:
        raw_response = _call_groq(prompt)
        json_obj = _extract_json_from_text(raw_response)

        if json_obj is None:
            logger.error("planner: failed to extract JSON from response")
            return None

        json_obj = _repair_plan_data(json_obj, command)

        plan_obj = ScenePlan.model_validate(json_obj)
        elapsed = time.perf_counter() - start
        logger.info("planner: latency=%.1fms scene_type=%s num_objects=%d", 
                    elapsed * 1000, plan_obj.scene_type, plan_obj.num_objects)
        return plan_obj

    except Exception as e:
        logger.error("planner: failed to generate plan: %s", e)
        return None
