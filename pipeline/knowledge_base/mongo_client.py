"""
MongoDB client for the HoloScript knowledge base.

MongoDB is the source of truth for concepts, synonyms, descriptions, and asset
metadata. Redis is only a fast asset-path cache in front of MongoDB.
"""
from __future__ import annotations

import os

from pymongo import MongoClient
from pymongo.collection import Collection

from pipeline.cache import cache_asset, get_cached_asset

_client: MongoClient | None = None
_collection: Collection | None = None


def _category_from_asset_src(asset_src: str | None) -> str:
    if not asset_src:
        return "abstract"
    parts = asset_src.strip("/").split("/")
    if len(parts) >= 3 and parts[-3:-1] == ["assets", "meshes"]:
        return parts[-2]
    return "abstract"


def _doc_from_cached_asset(name: str) -> dict | None:
    asset_src = get_cached_asset(name)
    if not asset_src:
        return None
    try:
        doc = get_collection().find_one({"asset_src": asset_src})
        if doc:
            return doc
    except Exception:
        pass
    concept_id = name.lower().strip()
    return {
        "_id": concept_id,
        "asset_id": concept_id,
        "asset_src": asset_src,
        "category": _category_from_asset_src(asset_src),
        "type": "object",
        "description": f"Cached asset for {concept_id}",
        "synonyms": [concept_id],
        "embedding": [],
    }


def _cache_doc_asset(name: str, doc: dict | None) -> None:
    if not doc:
        return
    asset_src = doc.get("asset_src")
    if isinstance(asset_src, str) and asset_src.strip():
        cache_asset(name, asset_src)


def get_collection() -> Collection:
    global _client, _collection
    if _collection is not None:
        return _collection
    uri = os.getenv("MONGODB_URI")
    db = os.getenv("MONGODB_DB", "holoscript")
    coll = os.getenv("MONGODB_COLLECTION", "knowledge_base")
    if not uri:
        raise RuntimeError("MONGODB_URI not set in .env")
    _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    _collection = _client[db][coll]
    return _collection


def lookup_concept(name: str) -> dict | None:
    cached = _doc_from_cached_asset(name)
    if cached:
        return cached
    try:
        doc = get_collection().find_one({"_id": name.lower().strip()})
        if not doc:
            doc = get_collection().find_one({"synonyms": name.lower().strip()})
        _cache_doc_asset(name, doc)
        return doc
    except Exception:
        return None


def lookup_synonym(alias: str) -> dict | None:
    cached = _doc_from_cached_asset(alias)
    if cached:
        return cached
    try:
        doc = get_collection().find_one({"synonyms": alias.lower().strip()})
        _cache_doc_asset(alias, doc)
        return doc
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
    concept_id = name.lower().strip()
    clean_synonyms: list[str] = []
    for value in [concept_id, *(synonyms or [])]:
        alias = str(value).lower().strip()
        if alias and alias not in clean_synonyms:
            clean_synonyms.append(alias)

    try:
        collection = get_collection()
        existing = collection.find_one({"asset_src": asset_src}) if asset_src else None
        target_id = existing["_id"] if existing else concept_id
        update = {
            "$set": {
                "asset_src": asset_src,
                "category": category,
                "type": asset_type,
                "description": description,
            },
            "$setOnInsert": {
                "_id": target_id,
                "asset_id": name,
                "embedding": [],
            },
            "$addToSet": {"synonyms": {"$each": clean_synonyms}},
        }
        collection.update_one({"_id": target_id}, update, upsert=True)
        if asset_src:
            cache_asset(concept_id, asset_src)
    except Exception:
        pass


def get_all_concepts() -> list[dict]:
    try:
        return list(
            get_collection().find(
                {},
                {
                    "_id": 1,
                    "asset_src": 1,
                    "category": 1,
                    "type": 1,
                    "asset_id": 1,
                    "synonyms": 1,
                    "description": 1,
                },
            )
        )
    except Exception:
        return []


if __name__ == "__main__":
    print(f"MongoDB concepts: {len(get_all_concepts())}")
