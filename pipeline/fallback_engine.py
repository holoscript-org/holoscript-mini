"""
Handles concepts that scored below the embedding threshold.

LEVEL 1 -- concept_map has entry           -> use it
LEVEL 2 -- synonym_map has entry           -> remap to known concept
LEVEL 3 -- compound word split             -> try sub-tokens
LEVEL 4 -- signal LLM                      -> return as unresolved
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_KB = Path(__file__).parent / "knowledge_base"
with (_KB / "concept_map.json").open(encoding="utf-8") as f:
	CONCEPT_MAP = json.load(f)
with (_KB / "synonym_map.json").open(encoding="utf-8") as f:
	SYNONYM_MAP = {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def _resolve_single(token: str) -> str | None:
	t = token.lower().strip()
	if t in CONCEPT_MAP:
		return t
	if t in SYNONYM_MAP:
		for target in SYNONYM_MAP[t]:
			if target in CONCEPT_MAP:
				return target
	for part in re.split(r"[-_\s]", t):
		if part in CONCEPT_MAP:
			return part
		if part in SYNONYM_MAP:
			for target in SYNONYM_MAP[part]:
				if target in CONCEPT_MAP:
					return target
	return None


def resolve_intent(intent: dict) -> tuple[dict, list[str]]:
	"""
	Resolves anything in intent that didn't make it through embedding.
	Returns (resolved_intent, unresolved_list).
	"""
	resolved = {"objects": [], "structures": [], "systems": [], "effects": []}
	unresolved: list[str] = []
	for bucket in ["objects", "structures", "systems", "effects"]:
		for concept in intent.get(bucket, []):
			mapped = _resolve_single(concept)
			if mapped:
				entry = CONCEPT_MAP[mapped]
				target = {
					"object": "objects",
					"structure": "structures",
					"system": "systems",
					"effect": "effects",
				}.get(entry["type"], bucket)
				if mapped not in resolved[target]:
					resolved[target].append(mapped)
			else:
				if concept not in unresolved:
					unresolved.append(concept)
	return resolved, unresolved
