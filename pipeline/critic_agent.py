"""
Critic + Fixer agents for scene quality control.

Architecture:
  critique_and_fix_loop(scene, transcript) → scene   (Stage 8, up to 3 iterations)
      loop:
        └─ critique_scene(...)  → issues list      (Gemini Flash, ~800 tokens)
        └─ fix_scene(...)       → corrected scene   (Gemini Flash, ~3500 tokens)
      breaks early as soon as critique_scene() returns no issues.

Both agents use Gemini Flash via Vertex AI (ADC — no API key required),
reached through the shared client in llm/gemini_client.py, with no fallback
to Groq for this stage (the critic is a quality-refinement pass, not a
required one — if Gemini/Vertex is unavailable, critique_scene() simply
returns no issues and the scene passes through unchanged).

The entire pipeline fails silently: if anything goes wrong, the original
scene passes through unchanged. All decisions are logged for observability.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from core.utils.logger import get_logger
from llm.gemini_client import call_gemini
from pipeline.events import OnEvent, make_emitter, COMPLETED, OUTPUT, STARTED

logger = get_logger("critic_agent")

_MODEL_CRITIC = "gemini-2.5-flash"

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


# ─── Gemini call ──────────────────────────────────────────────────────────────

def _call_flash(system_prompt: str, user_prompt: str, max_tokens: int) -> str | None:
    """Single Gemini Flash call via the shared Vertex AI client. Returns text
    or None on any failure (unavailable SDK/ADC, API error)."""
    model = os.getenv("GEMINI_CRITIC_MODEL", _MODEL_CRITIC)
    return call_gemini(
        model,
        user_prompt,
        system_prompt,
        temperature=0.1,
        max_output_tokens=max_tokens,
        thinking_budget=0,
    )


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

def critique_scene(scene: dict, transcript: str, client: Any = None) -> list[dict]:
    """
    Ask the critic to review the scene against the transcript.
    Returns a list of issue dicts (empty if verdict is OK or on any failure).

    `client` is accepted and ignored for backward compatibility with older
    call sites — the shared llm/gemini_client.py module manages its own
    cached client internally now.
    """
    user_prompt = (
        f"USER REQUEST: {json.dumps(transcript)}\n\n"
        f"SCENE JSON: {json.dumps(scene, separators=(',', ':'))}"
    )
    raw    = _call_flash(_CRITIC_SYSTEM, user_prompt, max_tokens=2000)
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
    client: Any = None,
    verified_assets: list[dict] | None = None,
) -> dict | None:
    """
    Ask the fixer to apply the listed issues to the scene.
    Returns the corrected scene dict or None if the fixer fails or returns garbage.

    `client` is accepted and ignored for backward compatibility — see
    critique_scene() docstring.
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
    raw    = _call_flash(_FIXER_SYSTEM, user_prompt, max_tokens=4000)
    parsed = _extract_json(raw)
    if not isinstance(parsed, dict):
        logger.debug("Fixer: malformed response")
        return None
    return parsed


# ─── Public entry points ──────────────────────────────────────────────────────

def critique_and_fix(
    scene: dict,
    transcript: str,
    verified_assets: list[dict] | None = None,
) -> dict:
    """
    Single critic → optional fixer pass (one iteration, no loop).

    Kept for any caller that wants a single-shot critique/fix rather than
    the iterative loop — critique_and_fix_loop() (below) is what
    pipeline_runner.py's Stage 8 actually uses.

    Always returns a valid scene dict — the original if anything fails.
    Failures are completely silent at the scene level; decisions are logged
    so you can see whether the critic is firing and what it catches.
    """
    allowed_paths = {a["path"] for a in (verified_assets or [])}
    try:
        issues = critique_scene(scene, transcript)

        if not issues:
            logger.info("Critic: verdict OK")
            return scene

        categories = [i.get("category", "?") for i in issues]
        logger.info("Critic: %d issue(s) → running fixer  %s", len(issues), categories)

        fixed = fix_scene(scene, issues, verified_assets=verified_assets)
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


def critique_and_fix_loop(
    scene: dict,
    transcript: str,
    verified_assets: list[dict] | None = None,
    max_iterations: int = 3,
    run_id: str = "",
    on_event: OnEvent | None = None,
) -> dict:
    """
    Iterative generate -> critique -> fix -> re-critique loop, capped at
    `max_iterations` (default 3, matching pipeline/repair_loop.py's own
    max_iterations=3 convention). Short-circuits as soon as a critique pass
    finds no issues.

    Each iteration is a separately-numbered, separately-timed pipeline event
    ("critic_iteration_1", "critic_iteration_2", ...) so the frontend can
    show live "iteration N of max_iterations, found K issues, fixing..."
    progress rather than a single opaque black-box stage.

    Always returns a valid scene dict — the last good scene if anything
    fails partway through. Never raises.
    """
    emit = make_emitter(run_id, on_event)
    allowed_paths = {a["path"] for a in (verified_assets or [])}
    current = scene

    for iteration in range(1, max_iterations + 1):
        stage_id = f"critic_iteration_{iteration}"
        label = f"Critic — Iteration {iteration}/{max_iterations}"
        emit(stage_id, STARTED, label)
        t0 = time.monotonic()

        try:
            issues = critique_scene(current, transcript)
        except Exception as exc:
            logger.warning("Critic iteration %d: critique failed: %s — stopping loop", iteration, exc)
            emit(stage_id, COMPLETED, label, elapsed_ms=int((time.monotonic() - t0) * 1000))
            break

        if not issues:
            logger.info("Critic iteration %d: verdict OK — loop complete", iteration)
            emit(stage_id, OUTPUT, label, payload={"iteration": iteration, "verdict": "OK", "issues": []})
            emit(stage_id, COMPLETED, label, elapsed_ms=int((time.monotonic() - t0) * 1000))
            break

        categories = [i.get("category", "?") for i in issues]
        logger.info("Critic iteration %d: %d issue(s) → running fixer  %s", iteration, len(issues), categories)
        emit(stage_id, OUTPUT, label, payload={"iteration": iteration, "verdict": "HAS_ISSUES", "issues": issues})

        try:
            fixed = fix_scene(current, issues, verified_assets=verified_assets)
        except Exception as exc:
            logger.warning("Critic iteration %d: fixer failed: %s — keeping previous scene", iteration, exc)
            emit(stage_id, COMPLETED, label, elapsed_ms=int((time.monotonic() - t0) * 1000))
            break

        if fixed and isinstance(fixed.get("objects"), list) and fixed["objects"]:
            fixed = _restore_meshes(current, fixed, allowed_paths)
            logger.info(
                "Critic iteration %d: fixer corrected scene (%d objects, was %d)",
                iteration, len(fixed["objects"]), len(current.get("objects", [])),
            )
            emit(stage_id, OUTPUT, label, payload={
                "iteration": iteration, "fixed": True,
                "object_count_before": len(current.get("objects", [])),
                "object_count_after": len(fixed["objects"]),
                "issues_addressed": issues,
            })
            current = fixed
        else:
            logger.warning("Critic iteration %d: fixer returned invalid/empty scene — stopping loop", iteration)
            emit(stage_id, COMPLETED, label, elapsed_ms=int((time.monotonic() - t0) * 1000))
            break

        emit(stage_id, COMPLETED, label, elapsed_ms=int((time.monotonic() - t0) * 1000))

    return current
