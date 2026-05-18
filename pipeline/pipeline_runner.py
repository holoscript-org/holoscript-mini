"""
Main entry point: transcript -> valid scene JSON -> core/outputs/live_scene.json

Usage:
	python -m pipeline.pipeline_runner "create a city on mars"
	python -m pipeline.pipeline_runner          # uses voice recorder
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
from pipeline.semantic_parser import SemanticParser
from pipeline.fallback_engine import resolve_intent
from pipeline.retrieval import retrieve
from pipeline.scene_builder import build_scene
from pipeline.llm_bridge import llm_generate_objects
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

print("[startup] Loading semantic parser (first run downloads model ~80MB)...")
_PARSER = SemanticParser()
print("[startup] Ready.")

logger = get_logger("pipeline")


def run_pipeline(transcript: str) -> dict:
	t0 = time.perf_counter()

	logger.info("Stage 1: transcript received (len=%d)", len(transcript or ""))

	stage_start = time.perf_counter()
	raw_intent = _PARSER.parse_intent(transcript)
	logger.info(
		"Stage 2: semantic parse (objects=%d, structures=%d, systems=%d, effects=%d, %dms)",
		len(raw_intent.get("objects", [])),
		len(raw_intent.get("structures", [])),
		len(raw_intent.get("systems", [])),
		len(raw_intent.get("effects", [])),
		int((time.perf_counter() - stage_start) * 1000),
	)

	stage_start = time.perf_counter()
	resolved_intent, unresolved = resolve_intent(raw_intent)
	logger.info(
		"Stage 3: fallback resolve (resolved=%d, unresolved=%d, %dms)",
		sum(len(resolved_intent.get(k, [])) for k in ["objects", "structures", "systems", "effects"]),
		len(unresolved),
		int((time.perf_counter() - stage_start) * 1000),
	)
	if unresolved:
		logger.debug("Unresolved concepts: %s", unresolved)

	has_concepts = any(resolved_intent[k] for k in ["objects", "structures", "systems"])

	if not has_concepts:
		logger.info("Stage 4: no concepts resolved; using LLM fallback")
		llm_objs = llm_generate_objects(transcript)
		scene = {
			"name": transcript[:50],
			"objects": llm_objs or [],
			"lights": DEFAULT_LIGHTS,
			"camera": {"position": [0, 5, 20], "target": [0, 0, 0], "fov": 65},
		}
		if not scene["objects"]:
			scene = DEMO_FALLBACK
		components = {"assets": [], "generators": [], "effects": []}
	else:
		stage_start = time.perf_counter()
		components = retrieve(resolved_intent)
		logger.info(
			"Stage 4: retrieval (assets=%d, generators=%d, effects=%d, %dms)",
			len(components.get("assets", [])),
			len(components.get("generators", [])),
			len(components.get("effects", [])),
			int((time.perf_counter() - stage_start) * 1000),
		)

		stage_start = time.perf_counter()
		scene = build_scene(components, resolved_intent)
		logger.info(
			"Stage 5: scene builder (objects=%d, %dms)",
			len(scene.get("objects", [])),
			int((time.perf_counter() - stage_start) * 1000),
		)

		if unresolved:
			logger.info("Stage 6: LLM for unresolved (%d concepts)", len(unresolved))
			for concept in unresolved:
				extra = llm_generate_objects(concept)
				if extra:
					scene["objects"].extend(extra)
			logger.info("Stage 6: unresolved LLM objects total=%d", len(scene.get("objects", [])))

	stage_start = time.perf_counter()
	scene = enhance_scene(transcript, resolved_intent, components, scene)
	logger.info(
		"Stage 7: scene enhancer (objects=%d, %dms)",
		len(scene.get("objects", [])),
		int((time.perf_counter() - stage_start) * 1000),
	)

	stage_start = time.perf_counter()
	vr = validate_scene(scene)
	logger.info(
		"Stage 8: validation (fatal=%s, errors=%d, %dms)",
		bool(vr.get("fatal")),
		len(vr.get("errors", [])),
		int((time.perf_counter() - stage_start) * 1000),
	)

	if vr.get("fatal"):
		logger.warning("Stage 9: repair (fatal) -> running repair loop")
		scene = repair(scene, [vr["fatal"]])
		vr = validate_scene(scene)

	if vr.get("errors"):
		logger.warning("Stage 9: repair (errors=%d) -> running repair loop", len(vr.get("errors", [])))
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

	print("\nRecording 5s -- SPEAK NOW")
	audio = record_audio(duration=5)
	transcript = transcribe(audio)
	if not transcript:
		print("No speech detected.")
		return DEMO_FALLBACK
	return run_pipeline(transcript)


if __name__ == "__main__":
	if len(sys.argv) > 1:
		run_pipeline(" ".join(sys.argv[1:]))
	else:
		run_with_voice()
