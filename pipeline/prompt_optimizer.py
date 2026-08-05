"""
Stage 2 — Prompt Optimizer.

Input:  the raw transcript (typed command, or a voice transcript that may
        carry ASR artifacts — filler words, false starts, homophones).
Output: {
    "optimized_prompt": str,        # one clear, complete scene description
    "clarifications_made": [str],   # short bullets, one per ambiguity resolved
    "original_prompt": str,         # echo of the input, for UI diffing
}

This stage does NOT invent scene content. Its only job is to turn an
ambiguous/fragmentary request into a clear one — resolving pronouns and
shorthand, making implicit structure explicit (e.g. "bouncing ball" implies
physics-driven motion, so downstream stages get an explicit cue) — while
preserving exactly what the user asked for. Refinement-style short commands
("add a red cube") are passed through nearly verbatim rather than padded out
into a full scene description.

Fails open: any error (missing SDK/credentials, malformed JSON, empty output)
returns the original transcript unchanged with no clarifications. This stage
must never block the pipeline — it's a quality-of-generation improvement, not
a required step.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from core.utils.logger import get_logger
from llm.gemini_client import call_llm
from pipeline.events import OnEvent, make_emitter, COMPLETED, OUTPUT, STARTED

logger = get_logger("prompt_optimizer")

_MODEL_GEMINI = "gemini-2.5-flash"
_MODEL_GROQ = "llama-3.3-70b-versatile"

_SYSTEM = """\
You are a prompt-clarity assistant for a 3D/holographic scene generator.
You receive a raw user request — possibly a voice transcript with filler
words, false starts, or ASR mistakes, or a short typed fragment.

Your ONLY job is to rewrite it into one clear, complete, unambiguous
description of the scene to build. Rules:
- Do NOT add objects, features, or details the user did not ask for or
  clearly imply. Never invent content.
- Resolve pronouns, sentence fragments, and shorthand (e.g. "solar system
  w/ rings" -> "a solar system scene including Saturn's rings").
- Make implicit structure explicit when it helps downstream generation
  (e.g. "bouncing ball" -> mention it should use physics-driven bouncing
  motion), but only when it's a natural implication, not an invention.
- If the request is a short refinement command (e.g. "add a red cube",
  "make it bigger", "move the sun left"), keep it almost verbatim — do not
  pad a short refinement into a full scene description.
- If the request is already clear, return it essentially unchanged.

Respond with ONLY this JSON (no markdown, no explanation):
{
  "optimized_prompt": "the rewritten request",
  "clarifications_made": ["short bullet per ambiguity resolved or assumption made"]
}
If nothing needed clarifying, return the original text as optimized_prompt
and an empty clarifications_made list.
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


def optimize(
    transcript: str,
    run_id: str = "",
    on_event: OnEvent | None = None,
) -> dict[str, Any]:
    """
    Run the prompt optimizer. Always returns a dict with optimized_prompt,
    clarifications_made, and original_prompt — falls back to the identity
    transform on any failure.
    """
    emit = make_emitter(run_id, on_event)
    emit("prompt_optimizer", STARTED, "Prompt Optimizer")
    t0 = time.monotonic()

    fallback = {
        "optimized_prompt": transcript,
        "clarifications_made": [],
        "original_prompt": transcript,
    }

    if not transcript or not transcript.strip():
        emit("prompt_optimizer", COMPLETED, "Prompt Optimizer", elapsed_ms=0)
        return fallback

    raw, provider = call_llm(
        _MODEL_GEMINI, _MODEL_GROQ, transcript, _SYSTEM, temperature=0.3
    )
    parsed = _extract_json(raw)

    if not isinstance(parsed, dict) or not parsed.get("optimized_prompt"):
        logger.warning("Prompt optimizer: no usable output — passing transcript through unchanged")
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        emit("prompt_optimizer", OUTPUT, "Prompt Optimizer", payload=fallback)
        emit("prompt_optimizer", COMPLETED, "Prompt Optimizer", elapsed_ms=elapsed_ms)
        return fallback

    clarifications = parsed.get("clarifications_made", [])
    result = {
        "optimized_prompt": str(parsed["optimized_prompt"]),
        "clarifications_made": clarifications if isinstance(clarifications, list) else [],
        "original_prompt": transcript,
    }

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "Prompt optimizer: %d clarification(s) via %s",
        len(result["clarifications_made"]),
        provider or "none",
    )
    emit("prompt_optimizer", OUTPUT, "Prompt Optimizer", payload=result, provider=provider)
    emit("prompt_optimizer", COMPLETED, "Prompt Optimizer", elapsed_ms=elapsed_ms, provider=provider)
    return result
