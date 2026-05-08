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

with (_KB / "concept_map.json").open(encoding="utf-8") as f:
	CONCEPT_MAP: dict[str, dict[str, Any]] = json.load(f)


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


def retrieve(intent: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
	"""intent = output of semantic_parser.parse_intent()"""
	assets: list[dict[str, Any]] = []
	generators: list[dict[str, Any]] = []
	effects: list[dict[str, Any]] = []

	for concept in intent.get("objects", []):
		entry = CONCEPT_MAP.get(concept, {})
		asset_id = entry.get("asset_id")
		asset_src = entry.get("asset_src", "")
		sidecar = _load_sidecar(asset_id)
		if not _glb_on_disk(asset_src):
			asset_src, asset_id = None, None
		assets.append(
			{
				"concept": concept,
				"asset_id": asset_id,
				"asset_src": asset_src,
				"sidecar": sidecar,
				"category": entry.get("category", "abstract"),
				"is_generator": False,
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

	return {"assets": assets, "generators": generators, "effects": effects}
