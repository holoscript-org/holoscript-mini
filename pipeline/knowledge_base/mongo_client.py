"""
MongoDB client for the HoloScript knowledge base.
Replaces concept_map.json, concept_descriptions.json, synonym_map.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection

_client: MongoClient | None = None
_collection: Collection | None = None


def get_collection() -> Collection:
    global _client, _collection
    if _collection is not None:
        return _collection
    uri  = os.getenv("MONGODB_URI")
    db   = os.getenv("MONGODB_DB", "holoscript")
    coll = os.getenv("MONGODB_COLLECTION", "knowledge_base")
    if not uri:
        raise RuntimeError("MONGODB_URI not set in .env")
    _client     = MongoClient(uri, serverSelectionTimeoutMS=5000)
    _collection = _client[db][coll]
    return _collection


def lookup_concept(name: str) -> dict | None:
    try:
        return get_collection().find_one({"_id": name.lower().strip()})
    except Exception:
        return None


def lookup_synonym(alias: str) -> dict | None:
    try:
        return get_collection().find_one({"synonyms": alias.lower().strip()})
    except Exception:
        return None


def register_concept(
    name: str,
    asset_src: str | None,
    category: str,
    asset_type: str,
    description: str = "",
    synonyms: list[str] | None = None,
) -> None:
    doc = {
        "_id":         name.lower().strip(),
        "asset_id":    name,
        "asset_src":   asset_src,
        "category":    category,
        "type":        asset_type,
        "description": description,
        "synonyms":    synonyms or [],
        "embedding":   [],
    }
    try:
        get_collection().update_one({"_id": doc["_id"]}, {"$set": doc}, upsert=True)
    except Exception:
        pass


def get_all_concepts() -> list[dict]:
    try:
        return list(get_collection().find(
            {}, {"_id": 1, "asset_src": 1, "category": 1, "type": 1,
                 "asset_id": 1, "synonyms": 1, "description": 1}
        ))
    except Exception:
        return []


def migrate_json_to_mongo() -> None:
    """
    One-time migration: reads the three JSON files and pushes to MongoDB.
    Run once: python -m pipeline.knowledge_base.mongo_client
    """
    kb   = Path(__file__).parent
    cmap = json.loads((kb / "concept_map.json").read_text(encoding="utf-8"))
    desc_path = kb / "concept_descriptions.json"
    desc = json.loads(desc_path.read_text(encoding="utf-8")) if desc_path.exists() else {}
    syn_path = kb / "synonym_map.json"
    raw_syns = json.loads(syn_path.read_text(encoding="utf-8")) if syn_path.exists() else {}

    # Build reverse synonym lookup: canonical → [aliases]
    rev: dict[str, list[str]] = {}
    for alias, targets in raw_syns.items():
        if alias.startswith("_"):
            continue
        target_list = targets if isinstance(targets, list) else [targets]
        for canonical in target_list:
            rev.setdefault(canonical, []).append(alias)

    ops = []
    for concept, meta in cmap.items():
        doc = {
            "_id":         concept.lower().strip(),
            "asset_id":    meta.get("asset_id", concept),
            "asset_src":   meta.get("asset_src"),
            "category":    meta.get("category", "unknown"),
            "type":        meta.get("type", "object"),
            "description": desc.get(concept, ""),
            "synonyms":    rev.get(concept, []),
            "embedding":   [],
        }
        ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True))

    if ops:
        res = get_collection().bulk_write(ops)
        print(f"Migrated {res.upserted_count + res.modified_count} concepts to MongoDB.")
    else:
        print("No concepts found in concept_map.json.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    migrate_json_to_mongo()
