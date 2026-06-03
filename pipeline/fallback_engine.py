"""
Handles concepts that scored below the embedding threshold.

LEVEL 1 -- MongoDB has direct entry          -> use it
LEVEL 2 -- MongoDB synonym match             -> remap to known concept
LEVEL 3 -- compound word split               -> try sub-tokens
LEVEL 4 -- signal LLM                        -> return as unresolved

Falls back to JSON files if MongoDB is unavailable.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from pipeline.knowledge_base.mongo_client import lookup_concept, lookup_synonym
from pipeline.knowledge_base.embedder import semantic_search


# JSON fallback — used only if MongoDB is down
_KB = Path(__file__).parent / "knowledge_base"
_CONCEPT_MAP_FALLBACK: dict | None = None
_SYNONYM_MAP_FALLBACK: dict | None = None


def _get_json_concept_map() -> dict:
    global _CONCEPT_MAP_FALLBACK
    if _CONCEPT_MAP_FALLBACK is None:
        try:
            with (_KB / "concept_map.json").open(encoding="utf-8") as f:
                _CONCEPT_MAP_FALLBACK = json.load(f)
        except Exception:
            _CONCEPT_MAP_FALLBACK = {}
    return _CONCEPT_MAP_FALLBACK


def _get_json_synonym_map() -> dict:
    global _SYNONYM_MAP_FALLBACK
    if _SYNONYM_MAP_FALLBACK is None:
        try:
            with (_KB / "synonym_map.json").open(encoding="utf-8") as f:
                raw = json.load(f)
                _SYNONYM_MAP_FALLBACK = {k: v for k, v in raw.items() if not k.startswith("_")}
        except Exception:
            _SYNONYM_MAP_FALLBACK = {}
    return _SYNONYM_MAP_FALLBACK


def _lookup_with_fallback(token: str) -> dict | None:
    """Try MongoDB first; fall back to JSON on failure."""
    doc = lookup_concept(token)
    if doc:
        return doc
    # JSON fallback
    cmap = _get_json_concept_map()
    entry = cmap.get(token)
    if entry:
        return {"_id": token, **entry}
    return None


def _synonym_with_fallback(token: str) -> dict | None:
    """Try MongoDB synonym lookup; fall back to JSON synonym_map."""
    doc = lookup_synonym(token)
    if doc:
        return doc
    # JSON fallback
    smap = _get_json_synonym_map()
    cmap = _get_json_concept_map()
    targets = smap.get(token, [])
    if isinstance(targets, str):
        targets = [targets]
    for canonical in targets:
        entry = cmap.get(canonical)
        if entry:
            return {"_id": canonical, **entry}
    return None


def _resolve_single(token: str) -> str | None:
    t = token.lower().strip()

    # Level 1: direct match
    doc = _lookup_with_fallback(t)
    if doc:
        return doc["_id"]

    # Level 2: synonym match
    doc = _synonym_with_fallback(t)
    if doc:
        return doc["_id"]

    # Level 3: compound word split
    for part in re.split(r"[-_\s]", t):
        if not part:
            continue
        doc = _lookup_with_fallback(part)
        if doc:
            return doc["_id"]
        doc = _synonym_with_fallback(part)
        if doc:
            return doc["_id"]

    # Level 4: semantic similarity (handles unknown phrasings, plurals, metaphors)
    try:
        results = semantic_search(t, top_k=1)
        if results:
            return results[0]["_id"]
    except Exception:
        pass

    return None


def _resolve_phrase(phrase: str) -> str | None:
    p = phrase.lower().strip()
    doc = _lookup_with_fallback(p)
    if doc:
        return doc["_id"]
    doc = _synonym_with_fallback(p)
    if doc:
        return doc["_id"]
    # Semantic fallback for multi-word phrases
    try:
        results = semantic_search(p, top_k=1)
        if results:
            return results[0]["_id"]
    except Exception:
        pass
    return None


def _get_entry(concept_id: str) -> dict:
    """Retrieve full entry for a resolved concept id."""
    doc = _lookup_with_fallback(concept_id)
    if doc:
        return doc
    # JSON fallback
    cmap = _get_json_concept_map()
    entry = cmap.get(concept_id, {})
    return {"_id": concept_id, **entry}


def resolve_intent(intent: dict) -> tuple[dict, list[str]]:
    """
    Resolves anything in intent that didn't make it through embedding.
    Returns (resolved_intent, unresolved_list).
    """
    resolved = {"objects": [], "structures": [], "systems": [], "effects": []}
    unresolved: list[str] = []

    phrase_hits = intent.get("_phrase_hits", []) if isinstance(intent, dict) else []
    for phrase in phrase_hits:
        if not isinstance(phrase, str):
            continue
        mapped = _resolve_phrase(phrase)
        if mapped:
            entry = _get_entry(mapped)
            target = {
                "object":    "objects",
                "structure": "structures",
                "system":    "systems",
                "effect":    "effects",
            }.get(entry.get("type", ""), "objects")
            if mapped not in resolved[target]:
                resolved[target].append(mapped)
        else:
            if phrase not in unresolved:
                unresolved.append(phrase)

    for bucket in ["objects", "structures", "systems", "effects"]:
        for concept in intent.get(bucket, []):
            mapped = _resolve_single(concept)
            if mapped:
                entry = _get_entry(mapped)
                target = {
                    "object":    "objects",
                    "structure": "structures",
                    "system":    "systems",
                    "effect":    "effects",
                }.get(entry.get("type", ""), bucket)
                if mapped not in resolved[target]:
                    resolved[target].append(mapped)
            else:
                if concept not in unresolved:
                    unresolved.append(concept)

    return resolved, unresolved
