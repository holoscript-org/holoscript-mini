"""
Download free 3D assets into core/assets/meshes and HDRIs into core/assets/hdri.

This is the one-time setup step for the voice -> JSON -> scene pipeline:

    python -m pipeline.asset_ingester

The ingester writes one sidecar JSON per downloaded GLB into
pipeline/knowledge_base/assets. kb_builder.py later reads those sidecars to
build concept_map.json and concept_descriptions.json.
"""

from __future__ import annotations

import io
import json
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
_MESHES = _ROOT / "core" / "assets" / "meshes"
_HDRI = _ROOT / "core" / "assets" / "hdri"
_KB = Path(__file__).parent / "knowledge_base" / "assets"

load_dotenv(_ROOT / ".env")

MAX_BYTES = 10 * 1024 * 1024

POLYPIZZA_QUERIES = {
    "humans": ["person", "character", "human figure"],
    "vehicles": ["car", "vehicle", "rocket", "spaceship"],
    "buildings": ["building", "skyscraper", "house", "tower"],
    "trees": ["tree", "palm tree", "pine tree", "plant"],
    "planets": ["planet", "sphere", "moon", "asteroid"],
    "satellites": ["satellite", "spacecraft", "space station"],
    "abstract": ["crystal", "gem", "orb", "ring"],
}
POLYPIZZA_PER_QUERY = 3

KENNEY_PACKS = {
    "vehicles": "https://kenney.nl/assets/car-kit",
    "buildings": "https://kenney.nl/assets/city-kit-roads",
    "trees": "https://kenney.nl/assets/nature-kit",
    "abstract": "https://kenney.nl/assets/mini-dungeon",
}
KENNEY_MAX_PER_PACK = 6

GLTF_SAMPLES = {
    "humans": [
        "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/CesiumMan/glTF-Binary/CesiumMan.glb",
    ],
    "vehicles": [
        "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/ToyCar/glTF-Binary/ToyCar.glb",
    ],
    "abstract": [
        "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/Avocado/glTF-Binary/Avocado.glb",
        "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/Lantern/glTF-Binary/Lantern.glb",
    ],
}


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


def _download_file(url: str, dest: Path) -> bool:
    """Stream-download a file. Returns True on success. Enforces MAX_BYTES."""
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
                    print(f"  skip {dest.name}: over {MAX_BYTES // 1024 // 1024}MB limit")
                    return False
                f.write(chunk)
        print(f"  ok {dest.name} ({size // 1024}KB)")
        return True
    except Exception as exc:
        print(f"  failed {dest.name}: {exc}")
        dest.unlink(missing_ok=True)
        return False


def _write_sidecar(
    glb_path: Path,
    category: str,
    tags: list[str],
    author: str = "",
    license_str: str = "",
) -> None:
    _KB.mkdir(parents=True, exist_ok=True)
    clean_tags = list(dict.fromkeys(t.lower().strip() for t in tags if t and t.strip()))
    asset_id = f"{category}_{glb_path.stem}"
    sidecar = {
        "id": asset_id,
        "src": f"/assets/meshes/{category}/{glb_path.name}",
        "category": category,
        "tags": clean_tags,
        "author": author,
        "license": license_str,
    }
    (_KB / f"{asset_id}.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")


def _poly_download_url(item: dict[str, Any]) -> str:
    for key in ("Download", "download", "downloadUrl", "download_url", "glb"):
        value = item.get(key)
        if isinstance(value, str) and value.lower().endswith(".glb"):
            return value
    nested = _json_get(item, "Asset", "Url", default="")
    if isinstance(nested, str) and nested.lower().endswith(".glb"):
        return nested
    return ""


def _polypizza_headers() -> dict[str, str]:
    api_key = os.getenv("POLYPIZZA_API_KEY") or os.getenv("POLY_PIZZA_API_KEY")
    if not api_key:
        return {}
    return {"x-auth-token": api_key.strip()}


def fetch_polypizza(max_per_query: int = POLYPIZZA_PER_QUERY) -> None:
    print("\n-- Poly Pizza --")
    headers = _polypizza_headers()
    if not headers:
        print("  skipping Poly Pizza: set POLYPIZZA_API_KEY if the API requires authorization")
        return
    for category, queries in POLYPIZZA_QUERIES.items():
        dest_dir = _MESHES / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        for query in queries:
            try:
                resp = requests.get(
                    f"https://api.poly.pizza/v1.1/search/{query}",
                    headers=headers,
                    timeout=10,
                )
                resp.raise_for_status()
                payload = resp.json()
                results = payload.get("results", payload if isinstance(payload, list) else [])
            except Exception as exc:
                print(f"  Poly Pizza search '{query}': {exc}")
                continue

            for item in results[:max_per_query]:
                if not isinstance(item, dict):
                    continue
                dl_url = _poly_download_url(item)
                if not dl_url:
                    continue
                title = item.get("Title") or item.get("title") or query
                dest = dest_dir / _safe_filename(str(title))
                item_tags = item.get("Tags") or item.get("tags") or []
                if isinstance(item_tags, str):
                    item_tags = item_tags.split()
                tags = list(item_tags) + [query, category.rstrip("s")]
                author = (
                    _json_get(item, "Creator", "Username", default="")
                    or _json_get(item, "creator", "username", default="")
                    or "unknown"
                )
                license_str = str(item.get("License") or item.get("license") or "CC-BY")
                if dest.exists():
                    _write_sidecar(dest, category, tags, str(author), license_str)
                    continue
                if _download_file(dl_url, dest):
                    _write_sidecar(dest, category, tags, str(author), license_str)
                time.sleep(0.3)


def _resolve_kenney_download_url(asset_url: str) -> str:
    if asset_url.lower().endswith(".zip"):
        return asset_url
    resp = requests.get(asset_url, timeout=15)
    resp.raise_for_status()
    matches = re.findall(r"href=['\"]([^'\"]+\.zip)['\"]", resp.text, flags=re.IGNORECASE)
    for match in matches:
        if "kenney" in match.lower():
            return match if match.startswith("http") else f"https://kenney.nl{match}"
    return matches[0] if matches else asset_url


def fetch_kenney() -> None:
    print("\n-- Kenney --")
    for category, asset_url in KENNEY_PACKS.items():
        dest_dir = _MESHES / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            print(f"  Downloading {category} pack...")
            zip_url = _resolve_kenney_download_url(asset_url)
            resp = requests.get(zip_url, timeout=60)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
                glb_entries = [
                    name
                    for name in archive.namelist()
                    if name.lower().endswith(".glb") and "__MACOSX" not in name
                ]
                for entry in glb_entries[:KENNEY_MAX_PER_PACK]:
                    dest = dest_dir / Path(entry).name
                    stem = dest.stem.lower().replace("-", " ").replace("_", " ")
                    tags = [part for part in stem.split() if not part.isdigit()]
                    tags += [category, category.rstrip("s")]
                    if dest.exists():
                        _write_sidecar(dest, category, tags, "Kenney", "CC0")
                        continue
                    data = archive.read(entry)
                    if len(data) > MAX_BYTES:
                        print(f"  skip {dest.name}: over {MAX_BYTES // 1024 // 1024}MB limit")
                        continue
                    dest.write_bytes(data)
                    _write_sidecar(dest, category, tags, "Kenney", "CC0")
                    print(f"  ok {dest.name}")
        except Exception as exc:
            print(f"  failed Kenney {category}: {exc}")


def fetch_gltf_samples() -> None:
    print("\n-- glTF-Sample-Models --")
    for category, urls in GLTF_SAMPLES.items():
        dest_dir = _MESHES / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        for url in urls:
            dest = dest_dir / Path(url).name
            tags = [Path(url).stem.lower(), category, category.rstrip("s")]
            if dest.exists():
                _write_sidecar(dest, category, tags, "Khronos Group", "CC-BY 4.0")
                continue
            if _download_file(url, dest):
                _write_sidecar(dest, category, tags, "Khronos Group", "CC-BY 4.0")


def fetch_polyhaven_hdri(max_hdris: int = 3) -> None:
    print("\n-- Poly Haven HDRIs --")
    _HDRI.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get("https://api.polyhaven.com/assets?t=hdris", timeout=10)
        resp.raise_for_status()
        slugs = list(resp.json().keys())[: max_hdris * 3]
    except Exception as exc:
        print(f"  failed Poly Haven listing: {exc}")
        return

    count = 0
    for slug in slugs:
        if count >= max_hdris:
            break
        out = _HDRI / f"{slug}_1k.hdr"
        if out.exists():
            count += 1
            continue
        try:
            files_resp = requests.get(f"https://api.polyhaven.com/files/{slug}", timeout=10)
            files_resp.raise_for_status()
            files = files_resp.json()
            url = _json_get(files, "hdri", "1k", "hdr", "url", default="")
            if url and _download_file(url, out):
                count += 1
                time.sleep(0.5)
        except Exception as exc:
            print(f"  failed {slug}: {exc}")


def run_ingestion() -> None:
    print("=== Asset Ingestion ===")
    _MESHES.mkdir(parents=True, exist_ok=True)
    _HDRI.mkdir(parents=True, exist_ok=True)
    _KB.mkdir(parents=True, exist_ok=True)
    fetch_polypizza()
    fetch_kenney()
    fetch_gltf_samples()
    fetch_polyhaven_hdri()
    total = sum(1 for _ in _MESHES.rglob("*.glb"))
    sidecars = sum(1 for _ in _KB.glob("*.json"))
    hdris = sum(1 for _ in _HDRI.glob("*.hdr"))
    print(f"\n=== Done: {total} GLBs, {sidecars} sidecars, {hdris} HDRIs ===")
    print("Next step: python -m pipeline.kb_builder")


if __name__ == "__main__":
    run_ingestion()
