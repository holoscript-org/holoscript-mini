"""
Returns disk-verified GLB assets that are genuine semantic matches for a
given intent.

The knowledge base source is MongoDB, with Redis only as a path cache. Local
JSON knowledge-base files are intentionally not consulted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.knowledge_base.mongo_client import lookup_concept

_ROOT = Path(__file__).resolve().parents[1]


def _get_concept(name: str) -> dict[str, Any] | None:
    return lookup_concept(name)


def _glb_exists(src: str | None) -> bool:
    if not src:
        return False
    rel = src.lstrip("/")
    return (_ROOT / "core" / rel).exists()


def _direct_match_score(concept: str, entry: dict[str, Any]) -> float:
    asset_src = entry.get("asset_src", "") or ""
    asset_id = entry.get("asset_id", "") or ""

    stem = Path(asset_src).stem.lower().replace("-", "_").replace(" ", "_")
    id_stem = asset_id.lower().replace("-", "_").replace(" ", "_")
    id_bare = id_stem.split("_", 1)[-1] if "_" in id_stem else id_stem

    c = concept.lower().replace(" ", "_").replace("-", "_")

    if c == stem or c == id_bare:
        return 1.0
    if c in stem or stem in c:
        return 0.8
    if c in id_bare or id_bare in c:
        return 0.7
    words = [w for w in c.split("_") if len(w) > 2]
    if words and any(w in stem or w in id_bare for w in words):
        return 0.65
    return 0.0


def get_verified_assets(intent: dict[str, Any]) -> list[dict[str, str]]:
    """
    Returns {concept, path, label, score} for Mongo concepts whose GLB exists
    on disk and semantically matches the requested concept.
    """
    results: list[dict[str, str]] = []
    seen_paths: set[str] = set()

    all_concepts: list[str] = (
        intent.get("objects", [])
        + intent.get("structures", [])
        + intent.get("systems", [])
    )

    for concept in all_concepts:
        entry = _get_concept(concept) or {}

        if entry.get("type") in ("system", "effect"):
            continue

        asset_src = entry.get("asset_src") or ""
        if not _glb_exists(asset_src):
            continue

        score = _direct_match_score(concept, entry)
        if score < 0.65:
            continue

        if asset_src in seen_paths:
            continue
        seen_paths.add(asset_src)

        results.append(
            {
                "concept": concept,
                "path": asset_src,
                "label": concept.replace("_", " ").title(),
                "score": str(round(score, 2)),
            }
        )

    results.sort(key=lambda x: float(x["score"]), reverse=True)
    return results


def build_mesh_menu(verified: list[dict[str, str]]) -> str:
    if not verified:
        return "  (none - build everything from primitives)"
    lines = []
    for item in verified:
        lines.append(f'  - {item["label"]} -> "{item["path"]}"')
    return "\n".join(lines)
