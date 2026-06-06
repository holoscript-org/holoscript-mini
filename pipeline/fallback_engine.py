"""
Resolve parser output against the Mongo-backed knowledge base.

Resolution order:
  1. Direct Mongo/Redis concept lookup
  2. Mongo synonym lookup
  3. Compound token lookup
  4. Mongo embedding similarity
"""
from __future__ import annotations

import re
from pathlib import Path

from pipeline.knowledge_base.embedder import semantic_search
from pipeline.knowledge_base.mongo_client import lookup_concept, lookup_synonym


def _lookup_variants(token: str) -> list[str]:
    normalized = token.lower().strip()
    compact = re.sub(r"[^a-z0-9]+", "", normalized)
    variants = [normalized]
    if compact and compact not in variants:
        variants.append(compact)
    return variants


def _lookup(token: str) -> dict | None:
    return lookup_concept(token)


def _lookup_synonym(token: str) -> dict | None:
    return lookup_synonym(token)


def _canonical_id(doc: dict) -> str:
    asset_src = doc.get("asset_src")
    synonyms = {
        str(alias).lower().strip()
        for alias in doc.get("synonyms", [])
        if isinstance(alias, str)
    }
    if isinstance(asset_src, str) and asset_src.strip():
        stem = Path(asset_src).stem.lower().strip()
        if stem in synonyms:
            return stem
    return str(doc["_id"])


def _resolve_single(token: str) -> str | None:
    t = token.lower().strip()

    for variant in _lookup_variants(t):
        doc = _lookup(variant)
        if doc:
            return _canonical_id(doc)

        doc = _lookup_synonym(variant)
        if doc:
            return _canonical_id(doc)

    for part in re.split(r"[-_\s]", t):
        if not part:
            continue
        doc = _lookup(part)
        if doc:
            return _canonical_id(doc)
        doc = _lookup_synonym(part)
        if doc:
            return _canonical_id(doc)

    try:
        results = semantic_search(t, top_k=1)
        if results:
            return results[0]["_id"]
    except Exception:
        pass

    return None


def _resolve_phrase(phrase: str) -> str | None:
    p = phrase.lower().strip()
    for variant in _lookup_variants(p):
        doc = _lookup(variant)
        if doc:
            return _canonical_id(doc)
        doc = _lookup_synonym(variant)
        if doc:
            return _canonical_id(doc)
    try:
        results = semantic_search(p, top_k=1)
        if results:
            return results[0]["_id"]
    except Exception:
        pass
    return None


def _get_entry(concept_id: str) -> dict:
    return _lookup(concept_id) or {"_id": concept_id}


def _bucket_for_entry(entry: dict, default: str = "objects") -> str:
    return {
        "object": "objects",
        "structure": "structures",
        "system": "systems",
        "effect": "effects",
    }.get(entry.get("type", ""), default)


def resolve_intent(intent: dict) -> tuple[dict, list[str]]:
    """
    Resolves low-confidence and unresolved parser output.
    Returns (resolved_intent, unresolved_list).
    """
    resolved = {"objects": [], "structures": [], "systems": [], "effects": []}
    unresolved: list[str] = []
    suppressed_tokens: set[str] = set()

    phrase_candidates: list[str] = []
    if isinstance(intent, dict):
        for phrase in intent.get("_phrase_hits", []):
            if isinstance(phrase, str) and phrase not in phrase_candidates:
                phrase_candidates.append(phrase)
        for phrase in intent.get("_unresolved_tokens", []):
            if isinstance(phrase, str) and " " in phrase and phrase not in phrase_candidates:
                phrase_candidates.append(phrase)

    for phrase in phrase_candidates:
        mapped = _resolve_phrase(phrase)
        if mapped:
            entry = _get_entry(mapped)
            target = _bucket_for_entry(entry)
            if mapped not in resolved[target]:
                resolved[target].append(mapped)
            suppressed_tokens.update(
                token for token in re.split(r"[-_\s]", phrase.lower().strip()) if token
            )
        elif phrase not in unresolved:
            unresolved.append(phrase)

    for bucket in ["objects", "structures", "systems", "effects"]:
        for concept in intent.get(bucket, []):
            if not isinstance(concept, str):
                continue
            if concept.lower().strip() in suppressed_tokens:
                continue
            mapped = _resolve_single(concept)
            if mapped:
                entry = _get_entry(mapped)
                target = _bucket_for_entry(entry, bucket)
                if mapped not in resolved[target]:
                    resolved[target].append(mapped)
            elif concept not in unresolved:
                unresolved.append(concept)

    if isinstance(intent, dict):
        for token in intent.get("_unresolved_tokens", []):
            if not isinstance(token, str) or " " in token:
                continue
            if token.lower().strip() in suppressed_tokens:
                continue
            mapped = _resolve_single(token)
            if mapped:
                entry = _get_entry(mapped)
                target = _bucket_for_entry(entry)
                if mapped not in resolved[target]:
                    resolved[target].append(mapped)
            elif token not in unresolved:
                unresolved.append(token)

    return resolved, unresolved
