"""
Stage 3 — Structured Intent Extraction.

Input:  the optimized_prompt from Stage 2 (pipeline/prompt_optimizer.py).
Output: a Scene Intent IR — a structured intermediate representation of what
        the scene needs, used as advisory context by the scene architect
        (pipeline/scene_architect.py) in all three of its passes.

Why this is a separate LLM stage and not a reuse of pipeline/semantic_parser.py:
the semantic parser is an *asset-recall* tool — it embeds the transcript and
scores it against a MongoDB corpus of known concept phrases to find candidate
mesh/primitive matches. It has no ability to reason about spatial
relationships ("the moon orbits the earth"), counts ("eight planets"),
dynamics ("bouncing", "steady orbit"), or mood ("dark and vast"). Those are
language-understanding tasks that need a real reasoning pass, not corpus
similarity search. This stage runs entirely on the optimized prompt and does
NOT depend on Mongo/asset lookups, so it is placed before semantic parsing in
execution order even though asset resolution conceptually follows "what does
the user want" in the pipeline's stage numbering.

The Scene Intent IR is NEVER written to core/outputs/live_scene.json and
NEVER passed through pipeline/scene_validator.py — it is a transient
reasoning artifact, not part of the canonical scene schema. This is
deliberate: adding it to the schema would create a second Python/TypeScript
dual-maintenance obligation (mirroring the existing scene_validator.py /
gui/lib/sceneFactory.ts burden) for something that only the architect prompt
needs to see.

Fails open: on any error, returns an all-empty IR. Every downstream consumer
must treat every field as optional/absent-safe.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from core.utils.logger import get_logger
from llm.gemini_client import call_llm
from pipeline.events import OnEvent, make_emitter, COMPLETED, OUTPUT, STARTED

logger = get_logger("intent_extractor")

_MODEL_GEMINI = "gemini-2.5-pro"
_MODEL_GROQ = "llama-3.3-70b-versatile"

_SYSTEM = """\
You are a scene-intent analyst for a 3D/holographic scene generator. You
receive a clear scene description and must extract a structured
representation of what the scene needs — this will guide a downstream
3D-scene-building system, not describe the final JSON yourself.

Respond with ONLY this JSON (no markdown, no explanation):
{
  "scene_type": "astronomical|anatomical|mechanical|abstract|physics_demo|architectural|other",
  "objects": [
    {"concept": "earth", "count": 1, "role": "primary|secondary|detail", "notes": "orbits the sun; has one moon"}
  ],
  "spatial_relationships": [
    {"subject": "moon", "relation": "orbits", "object": "earth"}
  ],
  "dynamics": [
    {"target": "earth", "motion": "orbit|spin|physics|none", "notes": "steady, slower than inner planets"}
  ],
  "mood_style": {
    "lighting_mood": "short description",
    "descriptors": ["vast", "cold", "majestic"]
  },
  "educational_focus": "the underlying concept being taught, if any, else empty string",
  "explicit_user_constraints": ["hard constraints the user explicitly stated, e.g. 'must include Saturn's rings', 'no ground plane'"]
}

Rules:
- Only extract what is stated or clearly implied. Do not invent objects,
  relationships, or constraints not present in the description.
- "count" should reflect explicit or clearly implied plurality (e.g. "eight
  planets" -> count semantics per-object, "a solar system" implies the Sun
  plus planets as separate object entries).
- Keep notes short (one clause) — they're hints for a downstream builder, not
  prose.
- If a field has nothing to report, use an empty list/string/object for it —
  never omit a key.
"""

_EMPTY_IR: dict[str, Any] = {
    "scene_type": "other",
    "objects": [],
    "spatial_relationships": [],
    "dynamics": [],
    "mood_style": {},
    "educational_focus": "",
    "explicit_user_constraints": [],
}


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


def _coerce_ir(parsed: dict[str, Any]) -> dict[str, Any]:
    """Fill any missing/malformed keys with the empty-IR defaults so every
    downstream field access is safe without extra None-checks."""
    ir = dict(_EMPTY_IR)
    for key, default in _EMPTY_IR.items():
        value = parsed.get(key, default)
        if type(value) is type(default):
            ir[key] = value
    return ir


def extract(
    optimized_prompt: str,
    run_id: str = "",
    on_event: OnEvent | None = None,
) -> dict[str, Any]:
    """
    Run structured intent extraction. Always returns a Scene Intent IR dict
    (all-empty fallback on any failure) — never raises.
    """
    emit = make_emitter(run_id, on_event)
    emit("intent_extraction", STARTED, "Structured Intent Extraction")
    t0 = time.monotonic()

    if not optimized_prompt or not optimized_prompt.strip():
        emit("intent_extraction", COMPLETED, "Structured Intent Extraction", elapsed_ms=0)
        return dict(_EMPTY_IR)

    raw, provider = call_llm(
        _MODEL_GEMINI, _MODEL_GROQ, optimized_prompt, _SYSTEM, temperature=0.3
    )
    parsed = _extract_json(raw)
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    if not isinstance(parsed, dict):
        logger.warning("Intent extractor: no usable output — using empty IR")
        ir = dict(_EMPTY_IR)
        emit("intent_extraction", OUTPUT, "Structured Intent Extraction", payload=ir)
        emit("intent_extraction", COMPLETED, "Structured Intent Extraction", elapsed_ms=elapsed_ms)
        return ir

    ir = _coerce_ir(parsed)
    logger.info(
        "Intent extractor: scene_type=%s, %d object(s), %d relationship(s) via %s",
        ir["scene_type"], len(ir["objects"]), len(ir["spatial_relationships"]), provider or "none",
    )
    emit("intent_extraction", OUTPUT, "Structured Intent Extraction", payload=ir, provider=provider)
    emit("intent_extraction", COMPLETED, "Structured Intent Extraction", elapsed_ms=elapsed_ms, provider=provider)
    return ir
