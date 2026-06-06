"""
Live Poly Pizza search for concept-based asset expansion.

Downloaded GLBs are stored under core/assets/meshes, while all knowledge-base
metadata is stored in MongoDB and Redis. No local KB sidecars are written.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from pipeline.cache import get_cached_asset
from pipeline.knowledge_base.embedder import embed_concept
from pipeline.knowledge_base.mongo_client import lookup_concept, register_concept

_ROOT = Path(__file__).resolve().parents[1]
_MESHES = _ROOT / "core" / "assets" / "meshes"

load_dotenv(_ROOT / ".env")

MAX_BYTES = 10 * 1024 * 1024
POLYPIZZA_URL = "https://api.poly.pizza/v1.1/search/{query}"


def _safe_filename(name: str, suffix: str = ".glb") -> str:
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip().lower()).strip("._-")
    if not stem:
        stem = "asset"
    return stem[:64] + suffix


def _json_get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def _polypizza_headers() -> dict[str, str]:
    api_key = os.getenv("POLYPIZZA_API_KEY") or os.getenv("POLY_PIZZA_API_KEY")
    if not api_key:
        return {}
    return {"x-auth-token": api_key.strip()}


def _poly_download_url(item: dict[str, Any]) -> str:
    for key in ("Download", "download", "downloadUrl", "download_url", "glb"):
        value = item.get(key)
        if isinstance(value, str) and value.lower().endswith(".glb"):
            return value
    nested = _json_get(item, "Asset", "Url", default="")
    if isinstance(nested, str) and nested.lower().endswith(".glb"):
        return nested
    return ""


def _download_file(url: str, dest: Path) -> bool:
    try:
        resp = requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()
        size = 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            for chunk in resp.iter_content(8192):
                if not chunk:
                    continue
                size += len(chunk)
                if size > MAX_BYTES:
                    dest.unlink(missing_ok=True)
                    return False
                f.write(chunk)
        return True
    except Exception:
        dest.unlink(missing_ok=True)
        return False


def _unique_glb_path(category: str, filename: str) -> Path:
    dest_dir = _MESHES / category
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / filename
    if not path.exists():
        return path
    stem = path.stem
    for idx in range(1, 1000):
        candidate = dest_dir / f"{stem}_{idx}.glb"
        if not candidate.exists():
            return candidate
    return path


def _asset_doc(
    concept: str,
    asset_src: str,
    category: str,
    tags: list[str] | None = None,
    author: str | None = None,
    license_str: str | None = None,
) -> dict[str, Any]:
    clean_tags = []
    for value in [concept, *(tags or [])]:
        tag = str(value).lower().strip()
        if tag and tag not in clean_tags:
            clean_tags.append(tag)
    doc: dict[str, Any] = {
        "_id": concept.lower().strip(),
        "asset_id": f"{category}_{Path(asset_src).stem}",
        "asset_src": asset_src,
        "category": category,
        "type": "object",
        "description": f"Downloaded via Poly Pizza: {concept}",
        "synonyms": clean_tags,
    }
    if author:
        doc["author"] = author
    if license_str:
        doc["license"] = license_str
    return doc


def _disk_path(asset_src: str) -> Path:
    return _ROOT / "core" / asset_src.lstrip("/")


def _search_polypizza(query: str, headers: dict[str, str]) -> list[dict[str, Any]]:
    resp = requests.get(POLYPIZZA_URL.format(query=query), headers=headers, timeout=10)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("results", payload if isinstance(payload, list) else [])


def _remember_asset_query(
    query: str,
    asset_src: str,
    category: str,
    tags: list[str] | None = None,
    author: str | None = None,
    license_str: str | None = None,
    update_embedding: bool = False,
) -> dict[str, Any]:
    doc = _asset_doc(query, asset_src, category, tags, author, license_str)
    register_concept(
        name=str(query),
        asset_src=asset_src,
        category=category,
        asset_type="object",
        description=str(doc["description"]),
        synonyms=doc["synonyms"],
    )
    if update_embedding:
        embed_concept(str(query), str(doc["description"]))
    return lookup_concept(str(query)) or doc


def fetch_live_assets(
    concepts: list[dict[str, str]],
    max_per_concept: int = 1,
    delay_s: float = 0.2,
) -> list[dict[str, Any]]:
    """Fetch assets for concept tokens. Returns [{"concept": str, "sidecar": dict}, ...]."""
    headers = _polypizza_headers()
    if not headers:
        return []

    found: list[dict[str, Any]] = []
    for entry in concepts:
        query = entry.get("concept")
        if not query:
            continue
        category = entry.get("category") or "abstract"

        existing = lookup_concept(str(query))
        if existing and isinstance(existing.get("asset_src"), str) and _disk_path(existing["asset_src"]).exists():
            found.append({"concept": query, "sidecar": existing, "source": "mongo"})
            continue

        redis_path = get_cached_asset(str(query))
        if redis_path and _disk_path(redis_path).exists():
            doc = _remember_asset_query(str(query), redis_path, category)
            found.append({"concept": query, "sidecar": doc, "source": "redis_cache"})
            continue

        try:
            results = _search_polypizza(str(query), headers)
        except Exception:
            continue

        for item in results[:max_per_concept]:
            if not isinstance(item, dict):
                continue
            dl_url = _poly_download_url(item)
            if not dl_url:
                continue

            title = item.get("Title") or item.get("title") or query
            filename = _safe_filename(str(title))
            original_dest = _MESHES / category / filename

            item_tags = item.get("Tags") or item.get("tags") or []
            if isinstance(item_tags, str):
                item_tags = item_tags.split()
            tags = list(item_tags) + [str(query), category.rstrip("s")]
            author = (
                _json_get(item, "Creator", "Username", default="")
                or _json_get(item, "creator", "username", default="")
                or ""
            )
            license_str = str(item.get("License") or item.get("license") or "")

            if original_dest.exists():
                asset_src = f"/assets/meshes/{category}/{original_dest.name}"
                doc = _remember_asset_query(
                    str(query), asset_src, category, tags, author or None, license_str or None
                )
                found.append({"concept": query, "sidecar": doc, "source": "cache"})
                continue

            dest = _unique_glb_path(category, filename)
            if _download_file(dl_url, dest):
                asset_src = f"/assets/meshes/{category}/{dest.name}"
                doc = _remember_asset_query(
                    str(query),
                    asset_src,
                    category,
                    tags,
                    author or None,
                    license_str or None,
                    update_embedding=True,
                )
                found.append({"concept": query, "sidecar": doc, "source": "download"})

            if delay_s:
                time.sleep(delay_s)

    return found
