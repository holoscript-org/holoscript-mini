"""
Redis-backed cache for HoloScript.
Two layers:
  - asset cache:  concept name -> local GLB file path  (TTL = REDIS_ASSET_TTL)
  - scene cache:  MD5(command)  -> validated scene JSON (TTL = REDIS_SCENE_TTL)

If Redis is unavailable the system continues without caching — no crash.
"""
from __future__ import annotations

import hashlib
import json
import os

import redis

_redis: redis.Redis | None = None
_unavailable = False


def _get_client() -> redis.Redis | None:
    global _redis, _unavailable
    if _unavailable:
        return None
    if _redis is not None:
        return _redis
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        _redis = redis.from_url(url, decode_responses=True, socket_connect_timeout=2)
        _redis.ping()
        return _redis
    except Exception as e:
        _unavailable = True
        _redis = None
        return None


def _asset_ttl() -> int:
    return int(os.getenv("REDIS_ASSET_TTL", 604800))


def _scene_ttl() -> int:
    return int(os.getenv("REDIS_SCENE_TTL", 21600))


# ── Asset cache ───────────────────────────────────────────────────────────────

def get_cached_asset(concept: str) -> str | None:
    r = _get_client()
    if r is None:
        return None
    try:
        return r.get(f"asset:{concept.lower().strip()}")
    except Exception:
        return None


def cache_asset(concept: str, local_path: str) -> None:
    r = _get_client()
    if r is None:
        return
    try:
        r.setex(f"asset:{concept.lower().strip()}", _asset_ttl(), local_path)
    except Exception:
        pass


# ── Scene cache ───────────────────────────────────────────────────────────────

def _scene_key(command: str) -> str:
    h = hashlib.md5(command.strip().lower().encode()).hexdigest()
    return f"scene:{h}"


def get_cached_scene(command: str) -> dict | None:
    r = _get_client()
    if r is None:
        return None
    try:
        raw = r.get(_scene_key(command))
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


def cache_scene(command: str, scene: dict) -> None:
    r = _get_client()
    if r is None:
        return
    try:
        r.setex(_scene_key(command), _scene_ttl(), json.dumps(scene))
    except Exception:
        pass


def invalidate_scene(command: str) -> None:
    r = _get_client()
    if r is None:
        return
    try:
        r.delete(_scene_key(command))
    except Exception:
        pass
