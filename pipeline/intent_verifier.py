"""
Stage 8.5 — Intent Verifier.

This is distinct from pipeline/critic_agent.py's Stage 8 loop. The critic
checks the scene for *object-level* defects (missing lights, bad physics
params, overlapping positions, a wrong orbit center) and patches those
defects directly in the final scene JSON. The intent verifier instead asks
a holistic question: "does this finished scene, taken as a whole, actually
satisfy what the user asked for?" — comparing it against the *original*
request and the Scene Intent IR (pipeline/intent_extractor.py's structured
summary of objects/relationships/dynamics/mood), not against a checklist of
technical defects.

If the verifier decides the scene falls short (e.g. the user asked for "a
model of the solar system with all 8 planets" and only 3 planets exist; or
asked for "a calm underwater scene" and got harsh white lighting with no
blue tint), it does NOT fall back to a generic/primitive placeholder. It
identifies which of the three scene_architect.py passes is responsible
(layout — wrong objects/composition; detail — wrong per-object properties;
finish — wrong mood/lighting/camera) and returns a target pass + a concrete
feedback string. pipeline/pipeline_runner.py then calls
scene_architect.regenerate_pass() to re-run exactly that pass with the
feedback injected into its prompt, modifying and realigning the existing
scene rather than discarding it.

verify_and_realign() is the public entry point: runs the verifier, and if
it flags a mismatch, drives up to `max_rounds` regenerate-then-reverify
cycles (default 2 — this runs *after* the critic loop already ran, so the
combined worst case stays bounded). Fails open at every step: any error, or
a verifier that can't produce a usable verdict, leaves the scene unchanged.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from core.utils.logger import get_logger
from llm.gemini_client import call_gemini
from pipeline.events import OnEvent, make_emitter, COMPLETED, OUTPUT, STARTED

logger = get_logger("intent_verifier")

_MODEL_VERIFIER = "gemini-2.5-flash"

_VERIFIER_SYSTEM = """\
You are a scene-intent auditor for a 3D/holographic scene generator. You
receive the user's ORIGINAL request, a structured summary of what they
asked for (SCENE INTENT), and the FINAL generated scene JSON. Your job is
to judge whether the finished scene, taken as a whole, actually satisfies
what the user asked for — not to nitpick technical details (a separate
technical reviewer already checked those).

Ask yourself:
- Are all the objects the user explicitly asked for present, in roughly
  the right quantity? (e.g. "all 8 planets" but only 3 exist -> mismatch)
- Do the spatial relationships and dynamics match what was described?
  (e.g. "the moon should orbit the earth" but it doesn't orbit anything)
- Does the overall mood/lighting/camera framing match the described
  atmosphere? (e.g. "dark and moody" but the scene is brightly lit)
- Is anything the user explicitly asked NOT to include present anyway, or
  vice versa?

Do NOT flag things a technical reviewer already owns: individual light
intensities being slightly off, minor physics parameter tuning, small
positioning overlaps. Only flag it if a reasonable person looking at this
scene would say "that's not really what I asked for."

If the scene falls short, decide which ONE of these three stages is most
responsible and is where a fix should be targeted:
  "layout" — wrong objects exist, wrong count, wrong composition/count of
             satellites or parts, missing an object category entirely
  "detail" — objects exist and are roughly right, but per-object properties
             (animation, materials, physics correctness) don't match intent
  "finish" — objects and details are fine, but overall lighting/camera/mood
             doesn't match the described atmosphere

Respond with ONLY this JSON (no markdown, no explanation):
{
  "satisfies_intent": true | false,
  "gaps": ["short description of each way the scene falls short, if any"],
  "target_pass": "layout" | "detail" | "finish" | null,
  "feedback": "concrete, actionable instruction for what to change in the target pass — empty string if satisfies_intent is true"
}
If the scene satisfies the request: {"satisfies_intent": true, "gaps": [], "target_pass": null, "feedback": ""}
"""


def _extract_json(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        cleaned = text.strip()
        if "```" in cleaned:
            cleaned = re.sub(r"```[a-z]*\n?", "", cleaned).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start == -1 or end <= start:
            return None
        return json.loads(cleaned[start:end])
    except Exception:
        return None


def verify_intent(
    scene: dict[str, Any],
    original_prompt: str,
    intent_ir: dict[str, Any],
) -> dict[str, Any]:
    """
    Single verification call. Returns
    {"satisfies_intent": bool, "gaps": [...], "target_pass": str|None, "feedback": str}.
    Fails open — on any error, returns satisfies_intent=True so the pipeline
    never blocks on a broken verifier.
    """
    fallback = {"satisfies_intent": True, "gaps": [], "target_pass": None, "feedback": ""}

    prompt = (
        f"ORIGINAL REQUEST: {json.dumps(original_prompt)}\n\n"
        f"SCENE INTENT: {json.dumps(intent_ir, separators=(',', ':'))}\n\n"
        f"FINAL SCENE JSON: {json.dumps(scene, separators=(',', ':'))}"
    )

    raw = call_gemini(
        _MODEL_VERIFIER, prompt, _VERIFIER_SYSTEM,
        temperature=0.1, max_output_tokens=1200, thinking_budget=0,
    )
    parsed = _extract_json(raw)
    if not isinstance(parsed, dict):
        logger.debug("Intent verifier: malformed/no response — assuming satisfied")
        return fallback

    result = dict(fallback)
    result["satisfies_intent"] = bool(parsed.get("satisfies_intent", True))
    gaps = parsed.get("gaps", [])
    result["gaps"] = gaps if isinstance(gaps, list) else []
    target_pass = parsed.get("target_pass")
    result["target_pass"] = target_pass if target_pass in ("layout", "detail", "finish") else None
    result["feedback"] = str(parsed.get("feedback") or "")
    return result


def verify_and_realign(
    scene: dict[str, Any],
    original_prompt: str,
    intent_ir: dict[str, Any],
    verified_assets: list[dict[str, str]],
    max_rounds: int = 2,
    run_id: str = "",
    on_event: OnEvent | None = None,
) -> dict[str, Any]:
    """
    Public entry point for Stage 8.5. Verifies the scene against the
    original request; if it falls short, calls
    scene_architect.regenerate_pass() to modify/realign the specific pass
    responsible, then re-verifies, up to `max_rounds` times. Never falls
    back to a generic/primitive scene — if a round's regeneration fails,
    the scene from before that round is kept and the loop stops.

    Always returns a valid scene dict — the best one obtained. Never raises.
    """
    from pipeline.scene_architect import regenerate_pass

    emit = make_emitter(run_id, on_event)
    current = scene

    for round_num in range(1, max_rounds + 1):
        stage_id = f"intent_verify_round_{round_num}"
        label = f"Intent Verification — Round {round_num}/{max_rounds}"
        emit(stage_id, STARTED, label)
        t0 = time.monotonic()

        try:
            verdict = verify_intent(current, original_prompt, intent_ir)
        except Exception as exc:
            logger.warning("Intent verifier round %d: exception %s — keeping current scene", round_num, exc)
            emit(stage_id, COMPLETED, label, elapsed_ms=int((time.monotonic() - t0) * 1000))
            break

        emit(stage_id, OUTPUT, label, payload=verdict)

        if verdict["satisfies_intent"] or not verdict["target_pass"]:
            logger.info("Intent verifier round %d: satisfies_intent=True — done", round_num)
            emit(stage_id, COMPLETED, label, elapsed_ms=int((time.monotonic() - t0) * 1000))
            break

        logger.info(
            "Intent verifier round %d: gap(s) found, targeting '%s' pass: %s",
            round_num, verdict["target_pass"], verdict["gaps"],
        )

        try:
            realigned = regenerate_pass(
                verdict["target_pass"], current, original_prompt, intent_ir,
                verified_assets, verdict["feedback"], run_id=run_id, on_event=on_event,
            )
        except Exception as exc:
            logger.warning("Intent verifier round %d: regenerate_pass failed: %s — keeping current scene", round_num, exc)
            emit(stage_id, COMPLETED, label, elapsed_ms=int((time.monotonic() - t0) * 1000))
            break

        if not realigned or not realigned.get("objects"):
            logger.warning("Intent verifier round %d: regeneration produced nothing usable — keeping current scene", round_num)
            emit(stage_id, COMPLETED, label, elapsed_ms=int((time.monotonic() - t0) * 1000))
            break

        current = realigned
        emit(stage_id, COMPLETED, label, elapsed_ms=int((time.monotonic() - t0) * 1000))

    return current
