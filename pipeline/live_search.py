"""
Live Poly Pizza search for concept-based asset expansion.

Uses concept tokens (not full transcripts), caches assets locally,
and returns sidecar metadata for retrieval expansion.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from pipeline.cache import get_cached_asset, cache_asset
from pipeline.knowledge_base.mongo_client import register_concept
from pipeline.knowledge_base.embedder import embed_concept

_ROOT = Path(__file__).resolve().parents[1]
_MESHES = _ROOT / "core" / "assets" / "meshes"
_KB_ASSETS = Path(__file__).parent / "knowledge_base" / "assets"

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


def _write_sidecar(
    glb_path: Path,
    category: str,
    tags: list[str],
    author: str | None,
    license_str: str | None,
) -> dict[str, Any]:
    _KB_ASSETS.mkdir(parents=True, exist_ok=True)
    clean_tags = list(dict.fromkeys(t.lower().strip() for t in tags if t and t.strip()))
    asset_id = f"{category}_{glb_path.stem}"
    sidecar: dict[str, Any] = {
        "id": asset_id,
        "src": f"/assets/meshes/{category}/{glb_path.name}",
        "category": category,
    }
    if clean_tags:
        sidecar["tags"] = clean_tags
    if author:
        sidecar["author"] = author
    if license_str:
        sidecar["license"] = license_str

    (_KB_ASSETS / f"{asset_id}.json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8"
    )
    return sidecar


def _load_sidecar(asset_id: str) -> dict[str, Any] | None:
    path = _KB_ASSETS / f"{asset_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _cached_sidecar_for_query(query: str) -> dict[str, Any] | None:
    if not _KB_ASSETS.exists():
        return None
    q = query.lower().strip()
    for path in _KB_ASSETS.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        tags = data.get("tags")
        if isinstance(tags, list) and any(q == str(tag).lower().strip() for tag in tags):
            return data
        asset_id = str(data.get("id", "")).lower()
        src = str(data.get("src", "")).lower()
        if q and (q in asset_id or q in src):
            return data
    return None


def _sidecar_for_glb(glb_path: Path, category: str) -> dict[str, Any] | None:
    asset_id = f"{category}_{glb_path.stem}"
    existing = _load_sidecar(asset_id)
    if existing:
        return existing
    return _write_sidecar(glb_path, category, [], None, None)


def _search_polypizza(query: str, headers: dict[str, str]) -> list[dict[str, Any]]:
    resp = requests.get(POLYPIZZA_URL.format(query=query), headers=headers, timeout=10)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("results", payload if isinstance(payload, list) else [])


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
        # Redis cache check first
        redis_path = get_cached_asset(str(query))
        if redis_path:
            from pathlib import Path as _Path
            disk = _Path(redis_path) if _Path(redis_path).is_absolute() else _ROOT / "core" / redis_path.lstrip("/")
            if disk.exists():
                sidecar = _sidecar_for_glb(disk, category) or {}
                found.append({"concept": query, "sidecar": sidecar, "source": "redis_cache"})
                continue

        cached = _cached_sidecar_for_query(str(query))
        if isinstance(cached, dict):
            src = cached.get("src")
            if isinstance(src, str) and src.strip():
                rel = src.lstrip("/").replace("assets/meshes/", "")
                if (_ROOT / "core" / "assets" / "meshes" / rel).exists():
                    found.append({"concept": query, "sidecar": cached, "source": "cache"})
                    continue
        try:
            results = _search_polypizza(query, headers)
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
            dest = _unique_glb_path(category, filename)

            item_tags = item.get("Tags") or item.get("tags") or []
            if isinstance(item_tags, str):
                item_tags = item_tags.split()
            tags = list(item_tags) + [query, category.rstrip("s")]
            author = (
                _json_get(item, "Creator", "Username", default="")
                or _json_get(item, "creator", "username", default="")
                or ""
            )
            license_str = str(item.get("License") or item.get("license") or "")

            if dest.exists():
                sidecar = _sidecar_for_glb(dest, category)
                if sidecar:
                    found.append({"concept": query, "sidecar": sidecar, "source": "cache"})
                continue

            if _download_file(dl_url, dest):
                sidecar = _write_sidecar(
                    dest,
                    category,
                    tags,
                    author or None,
                    license_str or None,
                )
                found.append({"concept": query, "sidecar": sidecar, "source": "download"})
                # Cache path in Redis and register in MongoDB
                asset_src = sidecar.get("src", "")
                cache_asset(str(query), asset_src)
                description = f"Downloaded via Poly Pizza: {query}"
                register_concept(
                    name=str(query),
                    asset_src=asset_src,
                    category=category,
                    asset_type="mesh",
                    description=description,
                    synonyms=tags,
                )
                embed_concept(str(query), description)

            if delay_s:
                time.sleep(delay_s)

    return found
