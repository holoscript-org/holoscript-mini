"""
Returns disk-verified GLB assets that are genuine semantic matches for a
given intent.

The knowledge base source is MongoDB, with Redis only as a path cache. Local
JSON knowledge-base files are intentionally not consulted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.knowledge_base.mongo_client import lookup_concept, register_concept

_ROOT = Path(__file__).resolve().parents[1]
_MESHES_ROOT = _ROOT / "core" / "assets" / "meshes"


def _get_concept(name: str) -> dict[str, Any] | None:
    return lookup_concept(name)


def _glb_exists(src: str | None) -> bool:
    if not src:
        return False
    rel = src.lstrip("/")
    return (_ROOT / "core" / rel).exists()


def _find_glb_on_disk(concept: str) -> str | None:
    """
    Scan mesh folders for a GLB whose filename matches the concept.
    Returns the asset_src path (e.g. /assets/meshes/abstract/mermaid.glb) or None.
    Used as a fallback when MongoDB has no entry for a concept.
    """
    c = concept.lower().replace(" ", "_").replace("-", "_")
    best: tuple[int, str] | None = None  # (score, path)
    for folder in _MESHES_ROOT.iterdir():
        if not folder.is_dir():
            continue
        for glb in folder.glob("*.glb"):
            stem = glb.stem.lower().replace("-", "_").replace(" ", "_")
            # Score: exact match > concept in stem > stem in concept > word overlap
            if stem == c:
                score = 3
            elif c in stem or stem in c:
                score = 2
            else:
                words = [w for w in c.split("_") if len(w) > 2]
                score = 1 if words and any(w in stem for w in words) else 0
            if score > 0:
                rel = f"/assets/meshes/{folder.name}/{glb.name}"
                if best is None or score > best[0]:
                    best = (score, rel)
    return best[1] if best else None


def _register_disk_asset(concept: str, asset_src: str) -> None:
    """Register an on-disk GLB in MongoDB so future lookups find it."""
    try:
        register_concept(
            name=concept,
            asset_src=asset_src,
            category=Path(asset_src).parent.name,
            asset_type="object",
            description=f"On-disk asset auto-registered: {concept}",
            synonyms=[concept.lower()],
        )
    except Exception:
        pass


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

        # MongoDB miss or file gone — try finding a matching GLB on disk directly
        if not _glb_exists(asset_src):
            disk_path = _find_glb_on_disk(concept)
            if not disk_path:
                continue
            asset_src = disk_path
            _register_disk_asset(concept, asset_src)
            entry = {"asset_src": asset_src}

        score = _direct_match_score(concept, entry)
        if score < 0.65:
            # Disk fallback: if the file exists and name contains concept words, use it
            disk_path = _find_glb_on_disk(concept)
            if not disk_path:
                continue
            asset_src = disk_path
            _register_disk_asset(concept, asset_src)
            score = 0.7  # confirmed disk match, accept it

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
