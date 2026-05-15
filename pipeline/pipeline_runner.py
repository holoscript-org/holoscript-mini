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

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv()

from pipeline.semantic_parser import SemanticParser
from pipeline.fallback_engine import resolve_intent
from pipeline.retrieval import retrieve
from pipeline.scene_builder import build_scene
from pipeline.llm_bridge import llm_generate_objects
from pipeline.scene_enhancer import enhance_scene
from pipeline.scene_validator import is_valid, validate_scene
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


def run_pipeline(transcript: str) -> dict:
	import time

	t0 = time.perf_counter()

	print(f"\n[1] Transcript: {transcript!r}")

	raw_intent = _PARSER.parse_intent(transcript)
	print(f"[2] Embedding intent: {raw_intent}  ({(time.perf_counter() - t0) * 1000:.0f}ms)")

	resolved_intent, unresolved = resolve_intent(raw_intent)
	print(f"[3] Resolved: {resolved_intent}")
	if unresolved:
		print(f"    Unresolved: {unresolved}")

	has_concepts = any(resolved_intent[k] for k in ["objects", "structures", "systems"])

	if not has_concepts:
		print("[4] Nothing resolved -- full LLM fallback")
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
		components = retrieve(resolved_intent)
		scene = build_scene(components, resolved_intent)
		print(
			f"[4] Built: {len(scene.get('objects', []))} objects  ({(time.perf_counter() - t0) * 1000:.0f}ms)"
		)

		if unresolved:
			print(f"[5] LLM for unresolved: {unresolved}")
			for concept in unresolved:
				extra = llm_generate_objects(concept)
				if extra:
					scene["objects"].extend(extra)

	scene = enhance_scene(transcript, resolved_intent, components, scene)

	vr = validate_scene(scene)
	print(
		f"[6] Validation -- fatal: {vr.get('fatal')}, errors: {len(vr.get('errors', []))}  ({(time.perf_counter() - t0) * 1000:.0f}ms)"
	)

	if vr.get("fatal"):
		scene = repair(scene, [vr["fatal"]])
		vr = validate_scene(scene)

	if vr.get("errors"):
		scene = repair(scene, vr["errors"])
		vr = validate_scene(scene)

	final = vr.get("scene") or scene
	if not final.get("objects"):
		final = DEMO_FALLBACK

	final = _strip_none(final)

	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	OUTPUT_PATH.write_text(json.dumps(final, indent=2), encoding="utf-8")

	elapsed = (time.perf_counter() - t0) * 1000
	print(f"[7] Saved -> {OUTPUT_PATH}")
	print(
		f"    Scene: \"{final.get('name')}\" -- {len(final.get('objects', []))} objects -- {elapsed:.0f}ms total"
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
