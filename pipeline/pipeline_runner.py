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
from pipeline.cache import get_cached_scene, cache_scene
from pipeline.semantic_parser import get_parser
from pipeline.fallback_engine import resolve_intent
from pipeline.asset_registry import get_verified_assets
from pipeline.scene_architect import generate_scene as architect_generate
from pipeline.critic_agent import critique_and_fix
from pipeline.scene_validator import validate_scene
from pipeline.repair_loop import DEMO_FALLBACK, repair

# Legacy path (used as fallback if architect fails)
from pipeline.retrieval import retrieve
from pipeline.scene_builder import build_scene
from pipeline.llm_bridge import llm_suggest_search_terms
from pipeline.scene_enhancer import enhance_scene

OUTPUT_PATH = _ROOT / "core" / "outputs" / "live_scene.json"

logger = get_logger("pipeline")


def _print_stage(message: str) -> None:
	print(f"[pipeline] {message}")


def _ensure_parser_ready() -> None:
	from pipeline.semantic_parser import _parser_instance
	if _parser_instance is not None:
		return
	print("[startup] Loading semantic parser (first run downloads model ~80MB)...")
	get_parser()
	print("[startup] Ready.")


def _strip_none(value):
	if isinstance(value, dict):
		cleaned = {k: _strip_none(v) for k, v in value.items() if v is not None}
		return {k: v for k, v in cleaned.items() if v is not None}
	if isinstance(value, list):
		return [_strip_none(v) for v in value if v is not None]
	return value


# ---------------------------------------------------------------------------
# Legacy builder path (fallback when architect fails)
# ---------------------------------------------------------------------------

def _legacy_build(transcript: str, resolved_intent: dict, unresolved: list) -> dict:
	"""Old retrieval → builder → enhancer path. Used only if Groq architect fails."""
	components = retrieve(resolved_intent, extra_candidates=unresolved)
	has_concepts = any(resolved_intent.get(k) for k in ["objects", "structures", "systems"])

	if not has_concepts:
		hints = llm_suggest_search_terms(transcript) or {}
		suggested = hints.get("suggested_search_terms") if isinstance(hints, dict) else []
		components = retrieve(resolved_intent, extra_candidates=suggested or [])

	scene = build_scene(components, resolved_intent)
	scene = enhance_scene(transcript, resolved_intent, components, scene)
	return scene


# ---------------------------------------------------------------------------
# Poly Pizza helpers
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
	"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
	"is", "it", "me", "of", "on", "or", "show", "the", "to", "with",
	"you", "your", "give", "make", "create", "see", "let", "display",
})


def _extract_candidate_phrases(transcript: str, unresolved: list[str]) -> list[str]:
	"""
	Return ordered list of phrases to try on Poly Pizza.
	Priority: bigrams of adjacent unresolved words > individual unresolved words.
	Generic — works for any concept without hardcoding examples.
	"""
	import re
	words = [
		w for w in re.findall(r"[a-z]+", transcript.lower())
		if w not in _STOPWORDS and len(w) >= 3
	]
	unresolved_set = {t.lower().strip() for t in unresolved}

	phrases: list[str] = []

	# Try trigrams first, then bigrams (adjacent unresolved words)
	for n in (3, 2):
		for i in range(len(words) - n + 1):
			chunk = words[i : i + n]
			# Only if all words in the chunk are unresolved (KB didn't know them)
			if all(w in unresolved_set for w in chunk):
				phrase = " ".join(chunk)
				if phrase not in phrases:
					phrases.append(phrase)

	phrase_words = {
		word
		for phrase in phrases
		for word in phrase.split()
	}

	# Then individual unresolved words (fallback)
	for token in unresolved:
		t = token.strip().lower()
		if t in phrase_words:
			continue
		if t and len(t) >= 3 and t not in _STOPWORDS and t not in phrases:
			phrases.append(t)

	return phrases


def _build_live_candidates(
	resolved_intent: dict,
	unresolved: list[str],
	verified_assets: list[dict],
	transcript: str = "",
) -> list[dict[str, str]]:
	"""
	Return concept+category dicts to search on Poly Pizza.
	Prefers compound phrases (bigrams/trigrams) so "wrist watch" is searched
	as a unit rather than "wrist" and "watch" separately.
	"""
	already_covered = {a["concept"] for a in verified_assets}
	candidates: list[dict[str, str]] = []

	unresolved_phrases = _extract_candidate_phrases(transcript, unresolved)
	phrase_words = {
		word
		for phrase in unresolved_phrases
		if " " in phrase
		for word in phrase.split()
	}

	# Resolved objects that had no local GLB match
	for concept in resolved_intent.get("objects", []):
		if concept in phrase_words:
			continue
		if concept not in already_covered:
			candidates.append({"concept": concept, "category": "abstract"})

	# Compound + individual phrases from unresolved tokens
	for phrase in unresolved_phrases:
		if phrase not in already_covered:
			candidates.append({"concept": phrase, "category": "abstract"})

	# Deduplicate, cap at 4 to keep latency reasonable
	seen: set[str] = set()
	unique: list[dict[str, str]] = []
	for c in candidates:
		if c["concept"] not in seen:
			seen.add(c["concept"])
			unique.append(c)
	return unique[:4]


def _fetch_polypizza(candidates: list[dict[str, str]]) -> list[dict]:
	"""Call live_search.fetch_live_assets and convert Mongo docs to verified assets."""
	try:
		from pipeline.live_search import fetch_live_assets
		results = fetch_live_assets(candidates, max_per_concept=1)
	except Exception as exc:
		logger.warning("Poly Pizza fetch failed: %s", exc)
		return []

	assets: list[dict] = []
	for r in results:
		doc = r.get("sidecar") or {}
		src = doc.get("asset_src") or doc.get("src", "")
		if not src:
			continue
		# Verify the file actually landed on disk
		from pathlib import Path as _Path
		disk_path = _ROOT / "core" / src.lstrip("/")
		if not disk_path.exists():
			continue
		assets.append({
			"concept": r["concept"],
			"path":    src,
			"label":   r["concept"].replace("_", " ").title(),
			"score":   "1.0",
		})
		logger.info("Poly Pizza: downloaded '%s' → %s", r["concept"], src)
	return assets


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

_REFINE_PREFIXES = frozenset({
	"add", "remove", "delete", "change", "make it", "move", "rotate",
	"scale", "replace", "now", "also", "update", "rename", "swap",
})


def _is_refinement(transcript: str) -> bool:
	"""Heuristic: does the transcript look like a scene refinement rather than a new scene?"""
	t = transcript.lower().strip()
	return any(t.startswith(p) for p in _REFINE_PREFIXES)


def run_pipeline(transcript: str) -> dict:
	t0 = time.perf_counter()
	_ensure_parser_ready()

	# ── Redis scene cache — skip for refinement commands ─────────────────────
	if not _is_refinement(transcript):
		cached = get_cached_scene(transcript)
		if cached:
			logger.info("Redis cache HIT — serving cached scene")
			_print_stage("Cache HIT: serving from Redis")
			OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
			OUTPUT_PATH.write_text(__import__("json").dumps(cached, indent=2), encoding="utf-8")
			return cached

	# ── Stage 1: receive transcript ──────────────────────────────────────────
	logger.info("Stage 1: transcript received (len=%d)", len(transcript or ""))
	_print_stage("Stage 1: transcript received")

	# ── Stage 2: semantic parse ──────────────────────────────────────────────
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

	# ── Stage 3: resolve intent ──────────────────────────────────────────────
	stage_start = time.perf_counter()
	resolved_intent, unresolved = resolve_intent(raw_intent)
	extra_unresolved = raw_intent.get("_unresolved_tokens", []) if isinstance(raw_intent, dict) else []
	resolved_terms = {
		str(concept).lower().strip()
		for bucket in ["objects", "structures", "systems", "effects"]
		for concept in resolved_intent.get(bucket, [])
	}
	for token in extra_unresolved:
		if not isinstance(token, str):
			continue
		if token.lower().strip() in resolved_terms:
			continue
		if token not in unresolved:
			unresolved.append(token)
	logger.info(
		"Stage 3: resolve (resolved=%d, unresolved=%d, %dms)",
		sum(len(resolved_intent.get(k, [])) for k in ["objects", "structures", "systems", "effects"]),
		len(unresolved),
		int((time.perf_counter() - stage_start) * 1000),
	)
	_print_stage("Stage 3: resolve complete")

	# ── Stage 4: build disk-verified asset menu for Groq ────────────────────
	stage_start = time.perf_counter()
	verified_assets = get_verified_assets(resolved_intent)
	logger.info(
		"Stage 4: asset registry (%d verified GLBs, %dms)",
		len(verified_assets),
		int((time.perf_counter() - stage_start) * 1000),
	)
	_print_stage(f"Stage 4: asset registry ({len(verified_assets)} verified meshes)")

	# ── Stage 4b: Poly Pizza live search for unresolved concepts ────────────
	# When a concept (e.g. "mermaid") is not in the local knowledge base,
	# search Poly Pizza, download the GLB, and hand it to the architect.
	live_candidates = _build_live_candidates(resolved_intent, unresolved, verified_assets, transcript)
	if live_candidates:
		stage_start = time.perf_counter()
		live_assets = _fetch_polypizza(live_candidates)
		if live_assets:
			verified_assets = verified_assets + live_assets
			logger.info(
				"Stage 4b: Poly Pizza (+%d new meshes, %dms)",
				len(live_assets),
				int((time.perf_counter() - stage_start) * 1000),
			)
			_print_stage(f"Stage 4b: Poly Pizza ({len(live_assets)} downloaded)")
		else:
			logger.info("Stage 4b: Poly Pizza — no new meshes found")

	# ── Stage 5: Groq scene architect ────────────────────────────────────────
	stage_start = time.perf_counter()
	scene = architect_generate(transcript, resolved_intent, verified_assets)
	architect_ok = scene is not None and bool(scene.get("objects"))
	logger.info(
		"Stage 5: architect (%s, objects=%d, %dms)",
		"ok" if architect_ok else "failed — using legacy path",
		len(scene.get("objects", [])) if scene else 0,
		int((time.perf_counter() - stage_start) * 1000),
	)
	_print_stage(f"Stage 5: architect {'complete' if architect_ok else 'failed — falling back'}")

	# ── Stage 5b: legacy fallback if architect failed ────────────────────────
	if not architect_ok:
		stage_start = time.perf_counter()
		scene = _legacy_build(transcript, resolved_intent, unresolved)
		logger.info(
			"Stage 5b: legacy build (objects=%d, %dms)",
			len(scene.get("objects", [])),
			int((time.perf_counter() - stage_start) * 1000),
		)
		_print_stage("Stage 5b: legacy build complete")

	# ── Stage 5.5: critic + fixer pass (silent fail, Gemini Flash) ───────────
	# Reviews the scene against the transcript for intent, spatial, scale,
	# lighting, animation and physics issues.  Runs only when Gemini is
	# configured; falls through to the original scene on any failure.
	stage_start = time.perf_counter()
	scene = critique_and_fix(scene, transcript)
	logger.info(
		"Stage 5.5: critic/fixer complete (%dms)",
		int((time.perf_counter() - stage_start) * 1000),
	)
	_print_stage("Stage 5.5: critic/fixer complete")

	# ── Stage 6: validate ────────────────────────────────────────────────────
	stage_start = time.perf_counter()
	vr = validate_scene(scene)
	logger.info(
		"Stage 6: validation (fatal=%s, errors=%d, %dms)",
		bool(vr.get("fatal")),
		len(vr.get("errors", [])),
		int((time.perf_counter() - stage_start) * 1000),
	)
	_print_stage("Stage 6: validation complete")

	# ── Stage 7: repair if needed ────────────────────────────────────────────
	if vr.get("fatal"):
		logger.warning("Stage 7: repair (fatal) -> running repair loop")
		_print_stage("Stage 7: repair (fatal)")
		scene = repair(scene, [vr["fatal"]])
		vr = validate_scene(scene)

	if vr.get("errors"):
		logger.warning("Stage 7: repair (errors=%d)", len(vr.get("errors", [])))
		_print_stage(f"Stage 7: repair ({len(vr['errors'])} errors)")
		scene = repair(scene, vr["errors"])
		vr = validate_scene(scene)

	final = vr.get("scene") or scene

	# ── Cache the result in Redis (new scenes only, not refinements) ──────────
	if not _is_refinement(transcript) and final.get("objects"):
		cache_scene(transcript, final)
		logger.info("Redis: scene cached for transcript (len=%d)", len(transcript))

	# ── Stage 8: fallback if still empty ─────────────────────────────────────
	if not final.get("objects"):
		logger.warning("Stage 8: fallback -> DEMO_FALLBACK")
		final = DEMO_FALLBACK

	final = _strip_none(final)

	# ── Stage 9: save ────────────────────────────────────────────────────────
	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	OUTPUT_PATH.write_text(json.dumps(final, indent=2), encoding="utf-8")

	elapsed = (time.perf_counter() - t0) * 1000
	logger.info("Stage 9: saved -> %s", OUTPUT_PATH)
	logger.info(
		"Complete: scene='%s' objects=%d total=%dms",
		final.get("name"),
		len(final.get("objects", [])),
		int(elapsed),
	)
	_print_stage(f"Stage 9: saved -> {OUTPUT_PATH}")
	return final


# ---------------------------------------------------------------------------
# Voice entry point
# ---------------------------------------------------------------------------

def run_with_voice() -> dict:
	from voice.recorder import record_audio
	from voice.transcriber import transcribe

	print("\n[voice] Speak now. Recording stops after silence is detected.")
	audio = record_audio(duration=10, use_vad=True)
	transcript = transcribe(audio)
	if not transcript:
		print("[voice] No speech detected.")
		return DEMO_FALLBACK
	print(f"[voice] Transcript: {transcript!r}")
	confirm = input("[voice] Is this correct? (y/n): ").strip().lower()
	if confirm not in ("y", "yes"):
		print("[voice] Aborted.")
		return DEMO_FALLBACK
	return run_pipeline(transcript)


if __name__ == "__main__":
	if len(sys.argv) > 1:
		run_pipeline(" ".join(sys.argv[1:]))
	else:
		run_with_voice()
