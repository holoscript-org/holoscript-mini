"""
Build generated knowledge-base files from ingested asset sidecars.

Outputs:
  pipeline/knowledge_base/concept_map.json
  pipeline/knowledge_base/concept_descriptions.json

Both are generated. Do not edit them by hand; update sidecars/synonym_map or
static entries here, then rerun:

    python -m pipeline.kb_builder
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_KB = Path(__file__).parent / "knowledge_base"
_ASSETS = _KB / "assets"

_OUT_MAP = _KB / "concept_map.json"
_OUT_DESC = _KB / "concept_descriptions.json"

CATEGORY_TYPE = {
    "humans": "object",
    "vehicles": "object",
    "buildings": "object",
    "trees": "object",
    "planets": "object",
    "satellites": "object",
    "abstract": "object",
}

STATIC_ENTRIES = {
    "forest": {
        "type": "structure",
        "generator": "scatter",
        "asset_category": "trees",
        "count": [8, 16],
        "radius": 12,
    },
    "city": {
        "type": "structure",
        "generator": "grid",
        "asset_category": "buildings",
        "count": [9, 16],
        "spacing": 3.5,
    },
    "crowd": {
        "type": "structure",
        "generator": "scatter",
        "asset_category": "humans",
        "count": [6, 12],
        "radius": 8,
    },
    "fleet": {
        "type": "structure",
        "generator": "scatter",
        "asset_category": "vehicles",
        "count": [4, 8],
        "radius": 10,
    },
    "solar_system": {
        "type": "system",
        "generator": "orbit_cluster",
        "central_category": "planets",
        "satellite_category": "planets",
        "count": [4, 8],
    },
    "atom": {
        "type": "system",
        "generator": "orbit_cluster",
        "central_category": "abstract",
        "satellite_category": "abstract",
        "count": [3, 6],
    },
    "orbit": {"type": "effect", "handler": "anim_orbit"},
    "spin": {"type": "effect", "handler": "anim_spin"},
    "rotate": {"type": "effect", "handler": "anim_spin"},
    "revolve": {"type": "effect", "handler": "anim_orbit"},
    "float": {"type": "effect", "handler": "anim_spin"},
    "hover": {"type": "effect", "handler": "anim_spin"},
}

STATIC_DESCRIPTIONS = {
    "forest": "forest trees woodland grove jungle dense trees plants nature scattered",
    "city": "city skyline buildings skyscrapers urban downtown metropolis towers structures",
    "crowd": "crowd group people standing figures gathering many humans",
    "fleet": "fleet group vehicles cars ships aircraft formation",
    "solar_system": "solar system planets orbiting star sun celestial bodies",
    "atom": "atom nucleus electrons orbiting particles quantum",
    "orbit": "orbit orbiting flying around circling revolving moving ellipse",
    "spin": "spin spinning rotating turning whirling swirling rotation",
    "rotate": "rotating turning spinning revolving angle",
    "revolve": "revolving orbiting circling moving around",
    "float": "floating hovering drifting gentle movement",
    "hover": "hovering floating staying still gentle bob",
    "humans": "human person figure character walking standing people",
    "vehicles": "vehicle car truck rocket ship aircraft spacecraft transport",
    "buildings": "building house structure skyscraper tower architecture",
    "trees": "tree forest plant nature palm pine oak vegetation",
    "planets": "planet sphere world globe rocky terrain celestial body moon mars earth",
    "satellites": "satellite probe station spacecraft orbiting device",
    "abstract": "crystal gem ring torus abstract shape decorative object",
}


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _tokens(text: str) -> list[str]:
    return [word.lower() for word in text.replace("_", " ").replace("-", " ").split() if len(word) > 1]


def _reverse_synonym_map() -> dict[str, list[str]]:
    syn_path = _KB / "synonym_map.json"
    syn = _load_json(syn_path)
    if not isinstance(syn, dict):
        return {}
    reverse: dict[str, list[str]] = {}
    for word, targets in syn.items():
        if word.startswith("_") or not isinstance(targets, list):
            continue
        for target in targets:
            if isinstance(target, str):
                reverse.setdefault(target, []).append(word)
    return reverse


def _concept_aliases(category: str) -> list[str]:
    singular = category[:-1] if category.endswith("s") else category
    aliases = [category, singular]
    if category == "humans":
        aliases.extend(["human", "person", "character", "figure"])
    elif category == "vehicles":
        aliases.extend(["vehicle", "car", "truck"])
    elif category == "buildings":
        aliases.extend(["building", "house", "tower", "skyscraper", "structure"])
    elif category == "trees":
        aliases.extend(["tree", "plant", "forest"])
    elif category == "planets":
        aliases.extend(["planet", "world", "moon", "asteroid", "star"])
    elif category == "satellites":
        aliases.extend(["satellite", "probe", "station"])
    elif category == "abstract":
        aliases.extend(["abstract", "crystal", "gem", "ring"])
    return list(dict.fromkeys(aliases))


def _register_object_concept(concept_map: dict[str, dict], concept: str, meta: dict) -> None:
    category = str(meta.get("category") or "abstract")
    concept_type = CATEGORY_TYPE.get(category, "object")
    concept = concept.lower().strip()
    if not concept or concept.isdigit() or len(concept) < 2:
        return
    concept_map.setdefault(
        concept,
        {
            "type": concept_type,
            "asset_id": meta["id"],
            "asset_src": meta["src"],
            "category": category,
        },
    )


def build() -> tuple[dict[str, dict], dict[str, str]]:
    concept_map: dict[str, dict] = {}
    descriptions: dict[str, str] = {}
    reverse_syn = _reverse_synonym_map()

    if _ASSETS.exists():
        for sidecar_path in sorted(_ASSETS.glob("*.json")):
            meta = _load_json(sidecar_path)
            if not isinstance(meta, dict) or not meta.get("id") or not meta.get("src"):
                continue
            category = str(meta.get("category") or "abstract")
            tags = meta.get("tags") if isinstance(meta.get("tags"), list) else []

            for tag in tags:
                if isinstance(tag, str):
                    _register_object_concept(concept_map, tag, meta)
                    for part in _tokens(tag):
                        _register_object_concept(concept_map, part, meta)

            for alias in _concept_aliases(category):
                _register_object_concept(concept_map, alias, meta)

    for key, entry in STATIC_ENTRIES.items():
        concept_map[key] = entry

    for concept, entry in concept_map.items():
        parts: list[str] = [concept]

        base_desc = STATIC_DESCRIPTIONS.get(concept, "")
        if not base_desc:
            base_desc = STATIC_DESCRIPTIONS.get(str(entry.get("category", "")), "")
        if base_desc:
            parts.append(base_desc)

        parts.extend(reverse_syn.get(concept, []))

        asset_id = entry.get("asset_id")
        if asset_id:
            sidecar = _load_json(_ASSETS / f"{asset_id}.json")
            if isinstance(sidecar, dict):
                tags = sidecar.get("tags")
                if isinstance(tags, list):
                    parts.extend(str(tag) for tag in tags)
                parts.append(str(sidecar.get("category", "")))

        category = entry.get("category") or entry.get("asset_category")
        if isinstance(category, str):
            parts.extend(_concept_aliases(category))

        unique_words = list(dict.fromkeys(word for part in parts for word in _tokens(str(part))))
        descriptions[concept] = " ".join(unique_words)

    return concept_map, descriptions


def main() -> None:
    print("Building knowledge base from ingested assets...")
    concept_map, descriptions = build()
    _KB.mkdir(parents=True, exist_ok=True)
    _OUT_MAP.write_text(json.dumps(concept_map, indent=2), encoding="utf-8")
    _OUT_DESC.write_text(json.dumps(descriptions, indent=2), encoding="utf-8")

    print(f"  concept_map.json -> {len(concept_map)} entries")
    print(f"  concept_descriptions.json -> {len(descriptions)} descriptions")
    by_type: dict[str, int] = {}
    for entry in concept_map.values():
        entry_type = str(entry.get("type", "unknown"))
        by_type[entry_type] = by_type.get(entry_type, 0) + 1
    for entry_type, count in sorted(by_type.items()):
        print(f"    {entry_type}: {count}")
    print("Next step: python -m pipeline.semantic_parser \"test prompt\"")


if __name__ == "__main__":
    main()
