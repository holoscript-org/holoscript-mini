"""
Generates sentence-transformer embeddings for every knowledge-base concept
and stores them in MongoDB.  Enables semantic similarity search so that
unknown queries like "flying creature" can match "dragon" without a synonym.

Uses the same model as semantic_parser.py — no extra download needed.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.knowledge_base.mongo_client import get_collection

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        # low_cpu_mem_usage=False avoids meta-tensor issue on PyTorch >= 2.7
        _model = SentenceTransformer(model_name, model_kwargs={"low_cpu_mem_usage": False})
    return _model


def embed_text(text: str) -> list[float]:
    vec = _get_model().encode(text.strip(), convert_to_numpy=True, normalize_embeddings=True)
    return vec.tolist()


def index_all_concepts() -> None:
    """
    One-time (re-runnable): embed every concept's description and store in MongoDB.
    Run with: python -m pipeline.knowledge_base.embedder
    """
    coll = get_collection()
    docs = list(coll.find({}, {"_id": 1, "description": 1}))
    if not docs:
        print("No concepts found in MongoDB. Run migration first.")
        return

    print(f"Embedding {len(docs)} concepts...")
    model = _get_model()

    texts = [doc.get("description") or doc["_id"] for doc in docs]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    from pymongo import UpdateOne
    ops = [
        UpdateOne(
            {"_id": docs[i]["_id"]},
            {"$set": {"embedding": embeddings[i].tolist()}},
        )
        for i in range(len(docs))
    ]
    res = coll.bulk_write(ops)
    print(f"Done. Updated {res.modified_count} concepts with embeddings.")


def semantic_search(
    query: str,
    top_k: int | None = None,
    threshold: float | None = None,
) -> list[dict]:
    """
    Find the most semantically similar concepts to the query.
    Returns up to top_k results with cosine similarity >= threshold.
    Both values default to env vars so they can be tuned without code changes.
    """
    k         = top_k    if top_k    is not None else int(os.getenv("EMBEDDING_TOP_K", "3"))
    min_score = threshold if threshold is not None else float(os.getenv("EMBEDDING_SIMILARITY_THRESHOLD", "0.45"))

    query_vec = np.array(embed_text(query))

    coll = get_collection()
    docs = list(coll.find(
        {"embedding": {"$not": {"$size": 0}}},
        {"_id": 1, "embedding": 1, "asset_src": 1, "type": 1, "category": 1, "asset_id": 1},
    ))
    if not docs:
        return []

    scores = [float(np.dot(query_vec, np.array(d["embedding"]))) for d in docs]
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [d for score, d in ranked[:k] if score >= min_score]


def embed_concept(name: str, description: str = "") -> None:
    """
    Generate and store an embedding for a single concept in MongoDB.
    Call this immediately after register_concept() so new assets are
    searchable without waiting for a full re-index.
    """
    text = description.strip() if description.strip() else name
    embedding = embed_text(text)
    try:
        get_collection().update_one(
            {"_id": name.lower().strip()},
            {"$set": {"embedding": embedding}},
        )
    except Exception:
        pass


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    index_all_concepts()
