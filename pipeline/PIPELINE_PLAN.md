# HoloScript Mini Pipeline Upgrade Plan (2026-05-14)

## Phase 0 — Baseline & Guardrails
- Confirm GUI schema (gui/lib/sceneFactory.ts) is the single source of truth.
- Enforce omission of any missing/unknown fields (never set to null).
- Review current pipeline, voice, and llm flows to preserve working systems.
- Define enhancement constraints:
  - LLM enhancer refines the base scene; no full regeneration.
  - Do not remove deterministic objects unless validator requires it.
  - Preserve asset URLs, mesh references, and deterministic placements unless intentionally refined.

## Phase 1 — VAD + Whisper Integration (voice/)
- Add VAD segmentation (Silero or WebRTC) to gate Whisper transcription.
- Detect speech start/end and transcribe only speech chunks.
- Keep real-time responsiveness and avoid blocking UI.
- Preserve current Whisper behavior for non-VAD flow.

## Phase 2 — Hybrid Retrieval Expansion (pipeline/)
- Add pipeline/live_search.py for Poly Pizza concept search.
- Trigger live search only when:
  - Confidence is low, or
  - Retrieval diversity is insufficient, or
  - No local assets exist.
- Cache results locally, generate sidecars, and integrate with KB conventions.
- Keep local KB as primary source of truth.

## Phase 3 — Retrieval Ranking & Scoring (pipeline/)
- Add ranking with factors: semantic similarity, synonym overlap, tag overlap,
  category match, asset quality, metadata completeness.
- Include confidence and source metadata in retrieval outputs.

## Phase 4 — LLM Scene Enhancement Pass (pipeline/ + llm/)
- Add pipeline/scene_enhancer.py.
- Update prompts to enforce GUI schema and JSON-only output.
- Ensure enhancer omits any missing fields rather than emitting null values.
- Input to Groq: transcript, semantic intent, retrieved assets, base scene, schema rules.
- Preserve deterministic grounding; refine composition, lighting, atmosphere, animation,
  camera framing without breaking schema.
- Align or deprecate llm/scene_schema.py if inconsistent with GUI schema.

## Phase 5 — Validation + Repair + Runner Update (pipeline/)
- Update pipeline runner flow:
  transcript → semantic parse → fallback → retrieval → live expansion → generators →
  scene builder → scene enhancer → validator → repair → output
- Keep validation and repair mandatory post-enhancement.

## Phase 6 — Smoke Tests & Stabilization
- Minimal CLI tests for VAD path, live search trigger logic, and enhancer preservation rules.
- Verify deterministic outputs remain stable and GUI-schema compatible.
