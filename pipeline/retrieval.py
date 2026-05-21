"""
Maps resolved intent -> concrete component list.

Reads concept_map.json (generated). Checks whether GLB files actually exist
on disk before committing to mesh type -- gracefully falls back to primitives.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_MESHES = _ROOT / "core" / "assets" / "meshes"
_KB = Path(__file__).parent / "knowledge_base"
_ASSETS = _KB / "assets"

LIVE_SEARCH_MAX_PER_CONCEPT = 1
LIVE_SEARCH_MIN_UNIQUE = 2
PREFERRED_LIVE_CATEGORIES = {"anatomy", "medical", "scientific", "creatures", "vehicles"}

with (_KB / "concept_map.json").open(encoding="utf-8") as f:
	CONCEPT_MAP: dict[str, dict[str, Any]] = json.load(f)


def _load_sidecar_index() -> dict[str, dict[str, Any]]:
	index: dict[str, dict[str, Any]] = {}
	if not _ASSETS.exists():
		return index
	for path in _ASSETS.glob("*.json"):
		try:
			data = json.loads(path.read_text(encoding="utf-8"))
		except Exception:
			continue
		if not isinstance(data, dict):
			continue
		tags = data.get("tags")
		if not isinstance(tags, list):
			continue
		for tag in tags:
			if isinstance(tag, str) and tag.strip():
				index[tag.lower().strip()] = data
	return index


def _load_sidecar(asset_id: str | None) -> dict[str, Any] | None:
	if not asset_id:
		return None
	path = _ASSETS / f"{asset_id}.json"
	return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _first_asset_in_category(category: str | None) -> dict[str, Any] | None:
	"""Return sidecar for the first GLB found in a category folder."""
	if not category:
		return None
	cat_dir = _MESHES / category
	if not cat_dir.exists():
		return None
	for glb in sorted(cat_dir.glob("*.glb")):
		sid = _ASSETS / f"{glb.stem}.json"
		if sid.exists():
			return json.loads(sid.read_text(encoding="utf-8"))
	return None


def _glb_on_disk(src: str | None) -> bool:
	if not src:
		return False
	rel = src.lstrip("/").replace("assets/meshes/", "")
	return (_ROOT / "core" / "assets" / "meshes" / rel).exists()


def _needs_live_search(
	object_concepts: list[str],
	assets: list[dict[str, Any]],
	missing_objects: list[dict[str, str]],
) -> bool:
	if not object_concepts:
		return False
	for missing in missing_objects:
		category = str(missing.get("category") or "").lower()
		if category in PREFERRED_LIVE_CATEGORIES:
			return True
	with_asset = [item for item in assets if item.get("asset_src")]
	if not with_asset:
		return True
	unique_assets = {item.get("asset_id") for item in with_asset if item.get("asset_id")}
	if len(object_concepts) > 1 and len(unique_assets) < min(len(object_concepts), LIVE_SEARCH_MIN_UNIQUE):
		return True
	return False


def _dedupe_assets(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
	best: dict[str, dict[str, Any]] = {}
	for item in assets:
		concept = str(item.get("concept") or "")
		if not concept:
			continue
		current = best.get(concept)
		if current is None:
			best[concept] = item
			continue
		current_has_src = bool(current.get("asset_src"))
		item_has_src = bool(item.get("asset_src"))
		if item_has_src and not current_has_src:
			best[concept] = item
			continue
		if item_has_src == current_has_src:
			if float(item.get("confidence", 0)) > float(current.get("confidence", 0)):
				best[concept] = item
	return list(best.values())


def _asset_metadata(sidecar: dict[str, Any] | None) -> dict[str, Any]:
	if not isinstance(sidecar, dict):
		return {}
	meta: dict[str, Any] = {}
	if isinstance(sidecar.get("author"), str) and sidecar["author"].strip():
		meta["author"] = sidecar["author"].strip()
	if isinstance(sidecar.get("license"), str) and sidecar["license"].strip():
		meta["license"] = sidecar["license"].strip()
	return meta


def _score_asset(concept: str, category: str | None, sidecar: dict[str, Any] | None, asset_src: str | None) -> float:
	if not asset_src:
		return 0.1
	concept_l = concept.lower().strip()
	category_l = (category or "").lower().strip()
	tags = sidecar.get("tags") if isinstance(sidecar, dict) else None
	matched_tags = 0
	if isinstance(tags, list):
		matched_tags = sum(
			1 for tag in tags if isinstance(tag, str) and concept_l == tag.lower().strip()
		)
	meta = _asset_metadata(sidecar)
	meta_score = 0.0
	if meta.get("author"):
		meta_score += 0.05
	if meta.get("license"):
		meta_score += 0.05
	cat_score = 0.1 if category_l else 0.0
	tag_score = 0.3 if matched_tags > 0 else 0.0
	base = 0.45
	return round(min(1.0, base + tag_score + cat_score + meta_score), 3)


def retrieve(
	intent: dict[str, list[str]],
	extra_candidates: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
	"""intent = output of semantic_parser.parse_intent()"""
	assets: list[dict[str, Any]] = []
	generators: list[dict[str, Any]] = []
	effects: list[dict[str, Any]] = []
	missing_objects: list[dict[str, str]] = []
	sidecar_index = _load_sidecar_index()

	for concept in intent.get("objects", []):
		entry = CONCEPT_MAP.get(concept, {})
		asset_id = entry.get("asset_id")
		asset_src = entry.get("asset_src", "")
		sidecar = _load_sidecar(asset_id)
		source = "kb" if asset_src else None
		if not asset_src:
			cached = sidecar_index.get(concept.lower().strip())
			if isinstance(cached, dict):
				asset_id = cached.get("id")
				asset_src = cached.get("src", "")
				sidecar = cached
				source = "kb"
		if not _glb_on_disk(asset_src):
			asset_src, asset_id = None, None
			missing_objects.append(
				{
					"concept": concept,
					"category": entry.get("category") or "abstract",
				}
			)
			source = None
		category = entry.get("category", "abstract")
		confidence = _score_asset(concept, category, sidecar, asset_src)
		meta = _asset_metadata(sidecar)
		assets.append(
			{
				"concept": concept,
				"concept_id": concept,
				"asset_id": asset_id,
				"asset_src": asset_src,
				"sidecar": sidecar,
				"category": category,
				"is_generator": False,
				**({"confidence": confidence} if confidence else {}),
				**({"source": source} if source else {}),
				**({"tags": sidecar.get("tags")} if isinstance(sidecar, dict) and isinstance(sidecar.get("tags"), list) and sidecar.get("tags") else {}),
				**({"metadata": meta} if meta else {}),
			}
		)

	for concept in intent.get("structures", []):
		entry = CONCEPT_MAP.get(concept, {})
		cat = entry.get("asset_category") or entry.get("category")
		sid = _first_asset_in_category(cat) if cat else None
		src = (sid or {}).get("src")
		assets.append(
			{
				"concept": concept,
				"concept_id": concept,
				"asset_id": (sid or {}).get("id"),
				"asset_src": src if _glb_on_disk(src or "") else None,
				"sidecar": sid,
				"category": cat,
				"is_generator": True,
				"generator_type": entry.get("generator", "scatter"),
				"generator_count": entry.get("count", [4, 8]),
				"generator_radius": entry.get("radius", 10),
				"generator_spacing": entry.get("spacing", 3.0),
			}
		)

	for concept in intent.get("systems", []):
		entry = CONCEPT_MAP.get(concept, {})
		cc = entry.get("central_category", "abstract")
		sc = entry.get("satellite_category", "abstract")
		generators.append(
			{
				"concept": concept,
				"concept_id": concept,
				"generator_type": entry.get("generator", "orbit_cluster"),
				"central_sidecar": _first_asset_in_category(cc),
				"satellite_sidecar": _first_asset_in_category(sc),
				"count": entry.get("count", [4, 8]),
			}
		)

	for concept in intent.get("effects", []):
		entry = CONCEPT_MAP.get(concept, {})
		effects.append(
			{"concept": concept, "handler": entry.get("handler", "anim_orbit")}
		)

	extra_candidates = [c for c in (extra_candidates or []) if isinstance(c, str) and c.strip()]
	if _needs_live_search(intent.get("objects", []), assets, missing_objects) or extra_candidates:
		try:
			from pipeline.live_search import fetch_live_assets

			candidates = list(missing_objects)
			if not candidates and extra_candidates:
				candidates = [{"concept": extra, "category": "abstract"} for extra in extra_candidates]
			elif not candidates:
				candidates = [
					{"concept": concept, "category": CONCEPT_MAP.get(concept, {}).get("category") or "abstract"}
					for concept in intent.get("objects", [])
				]
			elif extra_candidates:
				for extra in extra_candidates:
					candidates.append({"concept": extra, "category": "abstract"})
			live_results = fetch_live_assets(
				candidates,
				max_per_concept=LIVE_SEARCH_MAX_PER_CONCEPT,
			)
			for result in live_results:
				sidecar = result.get("sidecar")
				if not isinstance(sidecar, dict):
					continue
				category = sidecar.get("category", "abstract")
				concept = result.get("concept") or category or "live"
				confidence = _score_asset(concept, category, sidecar, sidecar.get("src"))
				meta = _asset_metadata(sidecar)
				assets.append(
					{
						"concept": concept,
						"concept_id": concept,
						"asset_id": sidecar.get("id"),
						"asset_src": sidecar.get("src"),
						"sidecar": sidecar,
						"category": category,
						"is_generator": False,
						**({"confidence": confidence} if confidence else {}),
						"source": "live",
						**({"tags": sidecar.get("tags")} if isinstance(sidecar.get("tags"), list) and sidecar.get("tags") else {}),
						**({"metadata": meta} if meta else {}),
					}
				)
		except Exception:
			pass

	assets = _dedupe_assets(assets)
	assets.sort(key=lambda item: float(item.get("confidence", 0)), reverse=True)
	return {"assets": assets, "generators": generators, "effects": effects}
