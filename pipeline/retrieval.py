"""
Maps resolved intent to concrete components using MongoDB as the knowledge base.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.knowledge_base.mongo_client import get_all_concepts, lookup_concept

_ROOT = Path(__file__).resolve().parents[1]

LIVE_SEARCH_MAX_PER_CONCEPT = 1
LIVE_SEARCH_MIN_UNIQUE = 2
PREFERRED_LIVE_CATEGORIES = {"anatomy", "medical", "scientific", "creatures", "vehicles"}


def _glb_on_disk(src: str | None) -> bool:
    if not src:
        return False
    rel = src.lstrip("/")
    return (_ROOT / "core" / rel).exists()


def _first_asset_in_category(category: str | None) -> dict[str, Any] | None:
    if not category:
        return None
    for doc in get_all_concepts():
        if doc.get("category") != category:
            continue
        if doc.get("type") not in (None, "object"):
            continue
        if _glb_on_disk(doc.get("asset_src")):
            return doc
    return None


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
    return len(object_concepts) > 1 and len(unique_assets) < min(
        len(object_concepts), LIVE_SEARCH_MIN_UNIQUE
    )


def _dedupe_assets(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for item in assets:
        concept = str(item.get("concept") or "")
        if not concept:
            continue
        current = best.get(concept)
        if current is None or (
            bool(item.get("asset_src"))
            and not bool(current.get("asset_src"))
        ) or float(item.get("confidence", 0)) > float(current.get("confidence", 0)):
            best[concept] = item
    return list(best.values())


def _asset_metadata(doc: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(doc, dict):
        return {}
    meta: dict[str, Any] = {}
    for key in ("author", "license"):
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            meta[key] = value.strip()
    return meta


def _score_asset(concept: str, doc: dict[str, Any] | None, asset_src: str | None) -> float:
    if not asset_src:
        return 0.1
    concept_l = concept.lower().strip()
    tags = doc.get("synonyms") if isinstance(doc, dict) else None
    tag_score = 0.3 if isinstance(tags, list) and concept_l in [str(t).lower() for t in tags] else 0
    cat_score = 0.1 if isinstance(doc, dict) and doc.get("category") else 0
    return round(min(1.0, 0.45 + tag_score + cat_score), 3)


def _asset_item(concept: str, doc: dict[str, Any] | None, source: str | None = None) -> dict[str, Any]:
    doc = doc or {}
    asset_src = doc.get("asset_src") if _glb_on_disk(doc.get("asset_src")) else None
    confidence = _score_asset(concept, doc, asset_src)
    meta = _asset_metadata(doc)
    synonyms = doc.get("synonyms") if isinstance(doc.get("synonyms"), list) else []
    return {
        "concept": concept,
        "concept_id": doc.get("_id", concept),
        "asset_id": doc.get("asset_id"),
        "asset_src": asset_src,
        "sidecar": {
            "id": doc.get("asset_id"),
            "src": asset_src,
            "category": doc.get("category"),
            "tags": synonyms,
        } if asset_src else None,
        "category": doc.get("category", "abstract"),
        "is_generator": False,
        "confidence": confidence,
        **({"source": source} if source else {}),
        **({"tags": synonyms} if synonyms else {}),
        **({"metadata": meta} if meta else {}),
    }


def retrieve(
    intent: dict[str, list[str]],
    extra_candidates: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    assets: list[dict[str, Any]] = []
    generators: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    missing_objects: list[dict[str, str]] = []

    for concept in intent.get("objects", []):
        entry = lookup_concept(concept) or {}
        item = _asset_item(concept, entry, "mongo" if entry.get("asset_src") else None)
        if not item.get("asset_src"):
            missing_objects.append(
                {"concept": concept, "category": entry.get("category") or "abstract"}
            )
        assets.append(item)

    for concept in intent.get("structures", []):
        entry = lookup_concept(concept) or {}
        cat = entry.get("asset_category") or entry.get("category")
        asset_doc = _first_asset_in_category(cat)
        src = (asset_doc or {}).get("asset_src")
        assets.append(
            {
                "concept": concept,
                "concept_id": concept,
                "asset_id": (asset_doc or {}).get("asset_id"),
                "asset_src": src if _glb_on_disk(src) else None,
                "sidecar": {
                    "id": (asset_doc or {}).get("asset_id"),
                    "src": src,
                    "category": cat,
                } if src else None,
                "category": cat,
                "is_generator": True,
                "generator_type": entry.get("generator", "scatter"),
                "generator_count": entry.get("count", [4, 8]),
                "generator_radius": entry.get("radius", 10),
                "generator_spacing": entry.get("spacing", 3.0),
            }
        )

    for concept in intent.get("systems", []):
        entry = lookup_concept(concept) or {}
        central = _first_asset_in_category(entry.get("central_category", "abstract"))
        satellite = _first_asset_in_category(entry.get("satellite_category", "abstract"))
        generators.append(
            {
                "concept": concept,
                "concept_id": concept,
                "generator_type": entry.get("generator", "orbit_cluster"),
                "central_sidecar": {
                    "src": (central or {}).get("asset_src"),
                    "category": (central or {}).get("category"),
                } if central else None,
                "satellite_sidecar": {
                    "src": (satellite or {}).get("asset_src"),
                    "category": (satellite or {}).get("category"),
                } if satellite else None,
                "count": entry.get("count", [4, 8]),
            }
        )

    for concept in intent.get("effects", []):
        entry = lookup_concept(concept) or {}
        effects.append({"concept": concept, "handler": entry.get("handler", "anim_orbit")})

    extra_candidates = [c for c in (extra_candidates or []) if isinstance(c, str) and c.strip()]
    if _needs_live_search(intent.get("objects", []), assets, missing_objects) or extra_candidates:
        try:
            from pipeline.live_search import fetch_live_assets

            candidates = list(missing_objects)
            if extra_candidates:
                candidates.extend(
                    {"concept": extra, "category": "abstract"} for extra in extra_candidates
                )
            live_results = fetch_live_assets(
                candidates,
                max_per_concept=LIVE_SEARCH_MAX_PER_CONCEPT,
            )
            for result in live_results:
                doc = result.get("sidecar")
                if not isinstance(doc, dict):
                    continue
                concept = result.get("concept") or doc.get("_id") or "live"
                assets.append(_asset_item(str(concept), doc, "live"))
        except Exception:
            pass

    assets = _dedupe_assets(assets)
    assets.sort(key=lambda item: float(item.get("confidence", 0)), reverse=True)
    return {"assets": assets, "generators": generators, "effects": effects}
