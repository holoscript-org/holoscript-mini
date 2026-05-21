"""
Returns disk-verified GLB assets that are genuine semantic matches for a
given intent.  Category fallbacks (e.g., asteroid.glb used as a "planet"
or "star") are excluded so the scene architect will fall back to primitives
instead of placing a wrong mesh.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_MESHES = _ROOT / "core" / "assets" / "meshes"
_KB = Path(__file__).parent / "knowledge_base"

with (_KB / "concept_map.json").open(encoding="utf-8") as _f:
    _CONCEPT_MAP: dict[str, dict[str, Any]] = json.load(_f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _glb_exists(src: str | None) -> bool:
    if not src:
        return False
    rel = src.lstrip("/")
    return (_ROOT / "core" / rel).exists()


def _direct_match_score(concept: str, entry: dict[str, Any]) -> float:
    """
    Score how closely the asset name matches the concept.
    1.0 = exact stem match, 0.6 = partial, 0.0 = no match (category fallback).
    """
    asset_src = entry.get("asset_src", "") or ""
    asset_id  = entry.get("asset_id",  "") or ""

    stem    = Path(asset_src).stem.lower().replace("-", "_").replace(" ", "_")
    id_stem = asset_id.lower().replace("-", "_").replace(" ", "_")
    # Strip leading category prefix from asset_id  (e.g. "abstract_heart" → "heart")
    id_bare = id_stem.split("_", 1)[-1] if "_" in id_stem else id_stem

    c = concept.lower().replace(" ", "_").replace("-", "_")

    if c == stem or c == id_bare:
        return 1.0
    if c in stem or stem in c:
        return 0.8
    if c in id_bare or id_bare in c:
        return 0.7
    # Multi-word concepts: check if any meaningful word appears in the stem
    words = [w for w in c.split("_") if len(w) > 2]
    if words and any(w in stem or w in id_bare for w in words):
        return 0.65
    return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_verified_assets(intent: dict[str, Any]) -> list[dict[str, str]]:
    """
    Returns a list of { concept, path, label } for assets that:
      1. Exist on disk as a .glb or .gltf file
      2. Are a genuine semantic match (score ≥ 0.65) — not a category fallback
      3. The concept is an object (not a system/effect, which should use primitives)

    This list is passed to the scene architect as the ONLY allowed mesh paths.
    """
    results: list[dict[str, str]] = []
    seen_paths: set[str] = set()

    all_concepts: list[str] = (
        intent.get("objects", [])
        + intent.get("structures", [])
        + intent.get("systems", [])
    )

    for concept in all_concepts:
        entry = _CONCEPT_MAP.get(concept, {})

        # Systems and effects are always built with primitives/generators
        if entry.get("type") in ("system", "effect"):
            continue

        asset_src = entry.get("asset_src") or ""
        if not _glb_exists(asset_src):
            continue

        score = _direct_match_score(concept, entry)
        if score < 0.65:
            continue  # Generic category fallback — skip

        if asset_src in seen_paths:
            continue
        seen_paths.add(asset_src)

        results.append({
            "concept": concept,
            "path":    asset_src,
            "label":   concept.replace("_", " ").title(),
            "score":   str(round(score, 2)),
        })

    # Sort best matches first so the prompt is clean
    results.sort(key=lambda x: float(x["score"]), reverse=True)
    return results


def build_mesh_menu(verified: list[dict[str, str]]) -> str:
    """Human-readable mesh list for the Groq prompt."""
    if not verified:
        return "  (none — build everything from primitives)"
    lines = []
    for item in verified:
        lines.append(f'  • {item["label"]} → "{item["path"]}"')
    return "\n".join(lines)
