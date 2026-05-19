"""
Main entry point: transcript -> valid scene JSON -> core/outputs/live_scene.json

Usage:
	python -m pipeline.pipeline_runner "create a city on mars"
	python -m pipeline.pipeline_runner          # uses voice recorder
	python -m pipeline.preload_semantic_model   # preloads the semantic model
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import time

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv()

from core.utils.logger import get_logger
from pipeline.semantic_parser import get_parser
from pipeline.fallback_engine import resolve_intent
from pipeline.retrieval import retrieve
from pipeline.scene_builder import build_scene
from pipeline.llm_bridge import llm_suggest_search_terms
from pipeline.scene_enhancer import enhance_scene
from pipeline.scene_validator import validate_scene
from pipeline.repair_loop import DEMO_FALLBACK, repair

OUTPUT_PATH = _ROOT / "core" / "outputs" / "live_scene.json"

DEFAULT_LIGHTS = [
	{"type": "ambient", "intensity": 0.4, "color": "#ffffff"},
	{
		"type": "directional",
		"intensity": 1.2,
		"color": "#ffffff",
		"position": [10, 10, 10],
		"castShadow": True,
	},
]

logger = get_logger("pipeline")


def _print_stage(message: str) -> None:
	print(f"[pipeline] {message}")


def _ensure_parser_ready() -> None:
	# Lazy-load to allow preloading via pipeline.preload_semantic_model.
	print("[startup] Loading semantic parser (first run downloads model ~80MB)...")
	get_parser()
	print("[startup] Ready.")


def run_pipeline(transcript: str) -> dict:
	t0 = time.perf_counter()
	_ensure_parser_ready()

	logger.info("Stage 1: transcript received (len=%d)", len(transcript or ""))
	_print_stage("Stage 1: transcript received")

	# Parse transcript into semantic intent buckets.
	stage_start = time.perf_counter()
	raw_intent = get_parser().parse_intent(transcript)
	logger.info(
		"Stage 2: semantic parse (objects=%d, structures=%d, systems=%d, effects=%d, %dms)",
		len(raw_intent.get("objects", [])),
		len(raw_intent.get("structures", [])),
		len(raw_intent.get("systems", [])),
		len(raw_intent.get("effects", [])),
		int((time.perf_counter() - stage_start) * 1000),
	)
	_print_stage("Stage 2: semantic parse complete")

	# Fill/normalize intent and track unresolved concepts.
	stage_start = time.perf_counter()
	resolved_intent, unresolved = resolve_intent(raw_intent)
	additional_unresolved = raw_intent.get("_unresolved_tokens", []) if isinstance(raw_intent, dict) else []
	for token in additional_unresolved:
		if isinstance(token, str) and token not in unresolved:
			unresolved.append(token)
	logger.info(
		"Stage 3: fallback resolve (resolved=%d, unresolved=%d, %dms)",
		sum(len(resolved_intent.get(k, [])) for k in ["objects", "structures", "systems", "effects"]),
		len(unresolved),
		int((time.perf_counter() - stage_start) * 1000),
	)
	_print_stage("Stage 3: fallback resolve complete")
	if unresolved:
		logger.debug("Unresolved concepts: %s", unresolved)

	has_concepts = any(resolved_intent[k] for k in ["objects", "structures", "systems"])

	if not has_concepts:
		# No grounded concepts; ask LLM for search hints and attempt live retrieval.
		logger.info("Stage 4: no concepts resolved; using LLM search hints")
		_print_stage("Stage 4: LLM search hints (no grounded concepts)")
		hints = llm_suggest_search_terms(transcript) or {}
		suggested = hints.get("suggested_search_terms") if isinstance(hints, dict) else []
		components = retrieve(resolved_intent, extra_candidates=suggested or [])
		if components.get("assets"):
			scene = build_scene(components, resolved_intent)
		else:
			scene = DEMO_FALLBACK
	else:
		# Retrieve grounded assets and build a deterministic base scene.
		phrase_unresolved = [u for u in unresolved if isinstance(u, str) and " " in u]
		stage_start = time.perf_counter()
		components = retrieve(resolved_intent, extra_candidates=unresolved)
		logger.info(
			"Stage 4: retrieval (assets=%d, generators=%d, effects=%d, %dms)",
			len(components.get("assets", [])),
			len(components.get("generators", [])),
			len(components.get("effects", [])),
			int((time.perf_counter() - stage_start) * 1000),
		)
		_print_stage("Stage 4: retrieval complete")

		# If a phrase-level asset was found, prefer it and suppress component objects.
		if phrase_unresolved:
			phrase_hits = [
				item
				for item in components.get("assets", [])
				if isinstance(item.get("concept"), str)
				and item.get("concept") in phrase_unresolved
			]
			if phrase_hits:
				phrase_hits.sort(key=lambda item: float(item.get("confidence", 0)), reverse=True)
				primary = phrase_hits[0]
				components["assets"] = [primary]
				resolved_intent = dict(resolved_intent)
				resolved_intent["objects"] = [primary.get("concept")]

		stage_start = time.perf_counter()
		scene = build_scene(components, resolved_intent)
		logger.info(
			"Stage 5: scene builder (objects=%d, %dms)",
			len(scene.get("objects", [])),
			int((time.perf_counter() - stage_start) * 1000),
		)
		_print_stage("Stage 5: scene builder complete")

		if unresolved:
			# Use LLM only for retrieval hints, not object generation.
			logger.info("Stage 6: LLM search hints for unresolved (%d concepts)", len(unresolved))
			_print_stage(f"Stage 6: LLM search hints ({len(unresolved)} concepts)")
			suggestions: list[str] = []
			if not phrase_unresolved:
				for concept in unresolved:
					hints = llm_suggest_search_terms(concept) or {}
					terms = hints.get("suggested_search_terms") if isinstance(hints, dict) else []
					if isinstance(terms, list):
						concept_tokens = [t for t in concept.lower().split() if t]
						for term in terms:
							if not isinstance(term, str):
								continue
							term_l = term.lower()
							if not concept_tokens or any(tok in term_l for tok in concept_tokens):
								suggestions.append(term)
			if suggestions:
				components = retrieve(resolved_intent, extra_candidates=suggestions)
				scene = build_scene(components, resolved_intent)
			_print_stage("Stage 6: unresolved LLM hints complete")

	# Scene enhancement pass (Groq) if configured.
	base_ids = {obj.get("id") for obj in scene.get("objects", []) if isinstance(obj, dict)}
	stage_start = time.perf_counter()
	scene = enhance_scene(transcript, resolved_intent, components, scene)
	if base_ids:
		scene["objects"] = [
			obj
			for obj in scene.get("objects", [])
			if isinstance(obj, dict) and obj.get("id") in base_ids
		]
	logger.info(
		"Stage 7: scene enhancer (objects=%d, %dms)",
		len(scene.get("objects", [])),
		int((time.perf_counter() - stage_start) * 1000),
	)
	_print_stage("Stage 7: scene enhancer complete")

	# Validate against schema and repair if needed.
	stage_start = time.perf_counter()
	vr = validate_scene(scene)
	logger.info(
		"Stage 8: validation (fatal=%s, errors=%d, %dms)",
		bool(vr.get("fatal")),
		len(vr.get("errors", [])),
		int((time.perf_counter() - stage_start) * 1000),
	)
	_print_stage("Stage 8: validation complete")

	if vr.get("fatal"):
		logger.warning("Stage 9: repair (fatal) -> running repair loop")
		_print_stage("Stage 9: repair (fatal) running")
		scene = repair(scene, [vr["fatal"]])
		vr = validate_scene(scene)

	if vr.get("errors"):
		logger.warning("Stage 9: repair (errors=%d) -> running repair loop", len(vr.get("errors", [])))
		_print_stage("Stage 9: repair (errors) running")
		scene = repair(scene, vr["errors"])
		vr = validate_scene(scene)

	final = vr.get("scene") or scene
	if not final.get("objects"):
		logger.warning("Stage 10: output fallback -> DEMO_FALLBACK")
		final = DEMO_FALLBACK

	final = _strip_none(final)

	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	OUTPUT_PATH.write_text(json.dumps(final, indent=2), encoding="utf-8")

	elapsed = (time.perf_counter() - t0) * 1000
	logger.info("Stage 11: saved -> %s", OUTPUT_PATH)
	logger.info(
		"Complete: scene='%s' objects=%d total=%dms",
		final.get("name"),
		len(final.get("objects", [])),
		int(elapsed),
	)
	_print_stage(f"Stage 11: saved -> {OUTPUT_PATH}")
	return final


def _strip_none(value):
	if isinstance(value, dict):
		cleaned = {k: _strip_none(v) for k, v in value.items() if v is not None}
		return {k: v for k, v in cleaned.items() if v is not None}
	if isinstance(value, list):
		return [_strip_none(v) for v in value if v is not None]
	return value


def run_with_voice() -> dict:
	from voice.recorder import record_audio
	from voice.transcriber import transcribe

	# Voice capture uses VAD and stops on silence.
	print("\n[voice] Speak now. Recording stops after silence is detected.")
	audio = record_audio(duration=10, use_vad=True)
	transcript = transcribe(audio)
	if not transcript:
		print("[voice] No speech detected.")
		return DEMO_FALLBACK
	print(f"[voice] Transcript: {transcript!r}")
	confirm = input("[voice] Is this correct? (y/n): ").strip().lower()
	if confirm not in ("y", "yes"):
		print("[voice] Aborted. Please try again.")
		return DEMO_FALLBACK
	return run_pipeline(transcript)


if __name__ == "__main__":
	if len(sys.argv) > 1:
		run_pipeline(" ".join(sys.argv[1:]))
	else:
		run_with_voice()
