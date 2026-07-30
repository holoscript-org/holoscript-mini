"""
Critic + Fixer agents for scene quality control.

Architecture:
  critique_and_fix(scene, transcript) → scene
      └─ critique_scene(...)  → issues list      (Gemini Flash, ~800 tokens)
      └─ fix_scene(...)       → corrected scene   (Gemini Flash, ~3500 tokens)

Both agents use Gemini Flash via GEMINI_API_KEY (direct API, not Vertex AI).
The entire pipeline fails silently: if anything goes wrong, the original scene
passes through unchanged.  All decisions are logged for observability.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from core.utils.logger import get_logger

logger = get_logger("critic_agent")

# ─── Critic system prompt ─────────────────────────────────────────────────────

_CRITIC_SYSTEM = """\
You are a 3D scene quality reviewer for a holographic display system.
You receive the original user request and the generated scene JSON.
Review it for the following categories of problems.

INTENT MISMATCH
- Scene mood/atmosphere doesn't match the user's description
  (e.g. user said "midnight" but ambient light intensity is 0.8 — too bright)
- Key thematic elements absent (user said "solar system" but nothing orbits)
- Wrong environment tone (user said "underwater" but no blue/dark lighting)

SPATIAL COHERENCE
- Two or more objects at identical or near-identical positions (within 0.5 units)
  and they are NOT in a parent-child relationship
- Object references a parent ID that doesn't exist in the scene
- All objects at origin [0,0,0] — nothing is actually placed

SCALE COHERENCE
- Object sizes are implausible relative to each other given the scene context
  (a planet at scale 0.05 next to a car at scale 4.0 with no narrative reason)
- One object is 50x larger than all others with no narrative reason
  EXCEPTION: ignore objects with geometry.type == "plane" — floor/ground planes
  are intentionally large and must not trigger this check

LIGHTING
- No ambient light present — scene will be pitch black
- No directional light — flat, unlit appearance
- All lights have intensity 0 or near 0
- Light color completely wrong for the described scene mood

ANIMATION
- Orbit animation where center [x,y,z] is far from any actual object in the scene
- Spin axis is [0,0,0] — invalid, no rotation will occur
- Object orbiting a parent but orbit center doesn't match parent position

PHYSICS (only check if any object has animation.type == "physics")
- gravity type: g value inconsistent with the described environment
  (moon/space/asteroid scene → g should be ≤ 2.0, not 9.8)
- gravity/projectile type: floor_y is at or above the object's starting y position
  (object will never fall, or hits floor immediately at spawn)
- shm type: amplitude is more than 2x the largest scene dimension
- pendulum type: pivot.y is below or equal to the bob's starting y position
  (pendulum would swing upward — physically backwards)

CAMERA
- camera.target points to empty space far from all objects
- fov below 20 or above 150
- camera.position is identical to camera.target (zero-length view vector)

Report at most 5 issues — focus on the most impactful ones first.

Respond with ONLY this JSON (no markdown, no explanation):
{
  "verdict": "OK" | "HAS_ISSUES",
  "issues": [
    {
      "category": "INTENT_MISMATCH|SPATIAL|SCALE|LIGHTING|ANIMATION|PHYSICS|CAMERA",
      "objects": ["id1", "id2"],
      "description": "specific description of the problem",
      "fix": "specific instruction for what to change"
    }
  ]
}
If no issues: {"verdict": "OK", "issues": []}
"""

# ─── Fixer system prompt ──────────────────────────────────────────────────────

_FIXER_SYSTEM = """\
You are a 3D scene repair agent.
You receive a scene JSON and a list of specific issues with exact fix instructions.
Apply ONLY the fixes listed. Do not change anything not mentioned in the issues list.
Do not rewrite the scene. Do not add objects unless the fix explicitly says to add one.
Do not remove objects unless the fix explicitly says to remove one.
CRITICAL: If an object has "type":"mesh", you MUST keep it as type="mesh" with its
original "model" path unchanged. Never convert a mesh object to a primitive.
CRITICAL: Preserve the top-level "summary" and every object "label"/"description" exactly
as given. These are educational text shown to students — never drop, shorten, or rewrite
them unless a fix instruction explicitly says the explanation is factually wrong.
Respond with ONLY the corrected scene JSON. No markdown, no explanation.
"""


# ─── Gemini client ────────────────────────────────────────────────────────────

def _get_gemini_client() -> Any | None:
    """Return a Vertex AI genai client using ADC (no API key required)."""
    try:
        from google import genai
        from google.genai.types import HttpOptions
        return genai.Client(
            vertexai=True,
            project=os.getenv("GCP_PROJECT", "reportevaluator"),
            location=os.getenv("GCP_LOCATION", "us-central1"),
            http_options=HttpOptions(api_version="v1"),
        )
    except Exception as exc:
        logger.debug("google-genai Vertex AI client failed: %s", exc)
        return None


def _call_flash(client: Any, system_prompt: str, user_prompt: str, max_tokens: int) -> str | None:
    """Single Gemini Flash call via Vertex AI. Returns text or None on any failure."""
    try:
        from google.genai import types as genai_types
        model = os.getenv("GEMINI_CRITIC_MODEL", "gemini-2.5-flash")
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0.1,
                max_output_tokens=max_tokens,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text
    except Exception as exc:
        logger.warning("Gemini Flash call failed: %s", exc)
        return None


# ─── JSON helpers ─────────────────────────────────────────────────────────────

def _extract_json(text: str | None) -> dict | None:
    if not text:
        return None
    try:
        cleaned = text.strip()
        if "```" in cleaned:
            cleaned = re.sub(r"```[a-z]*\n?", "", cleaned).strip()
        # Find outermost JSON object
        start = cleaned.find("{")
        end   = cleaned.rfind("}") + 1
        if start == -1 or end <= start:
            return None
        return json.loads(cleaned[start:end])
    except Exception:
        return None


# ─── Core agent functions ─────────────────────────────────────────────────────

def critique_scene(scene: dict, transcript: str, client: Any) -> list[dict]:
    """
    Ask the critic to review the scene against the transcript.
    Returns a list of issue dicts (empty if verdict is OK or on any failure).
    """
    user_prompt = (
        f"USER REQUEST: {json.dumps(transcript)}\n\n"
        f"SCENE JSON: {json.dumps(scene, separators=(',', ':'))}"
    )
    raw    = _call_flash(client, _CRITIC_SYSTEM, user_prompt, max_tokens=2000)
    parsed = _extract_json(raw)
    if not isinstance(parsed, dict):
        logger.debug("Critic: malformed response — skipping")
        return []
    if parsed.get("verdict") != "HAS_ISSUES":
        return []
    issues = parsed.get("issues", [])
    return issues if isinstance(issues, list) else []


def _restore_meshes(original: dict, fixed: dict, allowed_paths: set[str]) -> dict:
    """
    Hard guarantee: if the fixer converted a verified mesh object to a primitive,
    restore it. We trust the disk-verified asset list over the LLM's intent fix.
    """
    if not allowed_paths:
        return fixed
    original_by_id = {
        o.get("id"): o
        for o in original.get("objects", [])
        if isinstance(o, dict)
    }
    fixed_objects = list(fixed.get("objects", []))
    for i, obj in enumerate(fixed_objects):
        if not isinstance(obj, dict):
            continue
        orig = original_by_id.get(obj.get("id"))
        if orig is None:
            continue
        if orig.get("type") == "mesh" and orig.get("model", "") in allowed_paths:
            if obj.get("type") != "mesh":
                fixed_objects[i] = {**obj, "type": "mesh", "model": orig["model"]}
                logger.info(
                    "Restored mesh '%s' → '%s' (fixer had converted to %s)",
                    obj.get("id"), orig["model"], obj.get("type"),
                )
    fixed["objects"] = fixed_objects
    return fixed


def fix_scene(
    scene: dict,
    issues: list[dict],
    client: Any,
    verified_assets: list[dict] | None = None,
) -> dict | None:
    """
    Ask the fixer to apply the listed issues to the scene.
    Returns the corrected scene dict or None if the fixer fails or returns garbage.
    """
    mesh_note = ""
    if verified_assets:
        paths = "\n".join(
            f'  - {a.get("label", a["concept"])} → "{a["path"]}"'
            for a in verified_assets
        )
        mesh_note = (
            f"\n\nAVAILABLE MESHES (these are the ONLY 3D model files on disk — "
            f"keep them as type=\"mesh\"; never convert to primitive):\n{paths}"
        )
    user_prompt = (
        f"SCENE JSON: {json.dumps(scene, separators=(',', ':'))}\n\n"
        f"ISSUES TO FIX: {json.dumps(issues, separators=(',', ':'))}"
        f"{mesh_note}"
    )
    raw    = _call_flash(client, _FIXER_SYSTEM, user_prompt, max_tokens=4000)
    parsed = _extract_json(raw)
    if not isinstance(parsed, dict):
        logger.debug("Fixer: malformed response")
        return None
    return parsed


# ─── Public entry point ───────────────────────────────────────────────────────

def critique_and_fix(
    scene: dict,
    transcript: str,
    verified_assets: list[dict] | None = None,
) -> dict:
    """
    Full critic → optional fixer pass.

    Always returns a valid scene dict — the original if anything fails.
    Failures are completely silent at the scene level; decisions are logged
    so you can see whether the critic is firing and what it catches.
    """
    allowed_paths = {a["path"] for a in (verified_assets or [])}
    try:
        client = _get_gemini_client()
        if client is None:
            logger.debug("Critic/fixer skipped: google-genai not installed or Vertex AI client failed")
            return scene

        issues = critique_scene(scene, transcript, client)

        if not issues:
            logger.info("Critic: verdict OK")
            return scene

        categories = [i.get("category", "?") for i in issues]
        logger.info("Critic: %d issue(s) → running fixer  %s", len(issues), categories)

        fixed = fix_scene(scene, issues, client, verified_assets)
        if fixed and isinstance(fixed.get("objects"), list) and fixed["objects"]:
            fixed = _restore_meshes(scene, fixed, allowed_paths)
            logger.info(
                "Fixer: scene corrected (%d objects, was %d)",
                len(fixed["objects"]),
                len(scene.get("objects", [])),
            )
            return fixed

        logger.warning("Fixer: returned invalid or empty scene — keeping original")
        return scene

    except Exception as exc:
        logger.warning("Critic/fixer pipeline exception: %s — keeping original scene", exc)
        return scene
