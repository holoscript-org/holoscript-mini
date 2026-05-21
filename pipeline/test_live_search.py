import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

import json

from pipeline.live_search import fetch_live_assets

_KB = _ROOT / "pipeline" / "knowledge_base"
_ASSETS = _KB / "assets"
_MESHES = _ROOT / "core" / "assets" / "meshes"



def main() -> None:
    print("=== Live Search Smoke Test ===")
    concepts = [
        {"concept": "ring", "category": "abstract"},
        {"concept": "tower", "category": "buildings"},
        {"concept": "rocketship", "category": "vehicles"},
    ]
    concept_map = json.loads((_KB / "concept_map.json").read_text(encoding="utf-8"))
    sidecar_entries: list[dict] = []
    for path in _ASSETS.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            sidecar_entries.append(data)
    api_concepts: list[dict[str, str]] = []

    for entry in concepts:
        concept = entry.get("concept")
        if not concept:
            continue
        mapped = concept_map.get(concept, {})
        asset_id = mapped.get("asset_id")
        asset_src = mapped.get("asset_src", "")
        sidecar_path = _ASSETS / f"{asset_id}.json" if asset_id else None
        mesh_path = None
        if asset_src:
            rel = asset_src.lstrip("/").replace("assets/meshes/", "")
            mesh_path = _MESHES / rel

        in_kb = False
        if sidecar_path and sidecar_path.exists() and mesh_path and mesh_path.exists():
            in_kb = True
        else:
            needle = concept.lower().strip()
            for cached in sidecar_entries:
                tags = cached.get("tags")
                tag_match = isinstance(tags, list) and any(
                    needle == str(tag).lower().strip() for tag in tags
                )
                asset_id = str(cached.get("id", "")).lower()
                src = str(cached.get("src", "")).lower()
                text_match = needle and (needle in asset_id or needle in src)
                if not (tag_match or text_match):
                    continue
                src_val = cached.get("src", "")
                if isinstance(src_val, str) and src_val.strip():
                    rel = src_val.lstrip("/").replace("assets/meshes/", "")
                    if (_MESHES / rel).exists():
                        in_kb = True
                        break
        status = "kb" if in_kb else "api"
        print(f"Concept '{concept}': {status}")
        if not in_kb:
            api_concepts.append(entry)

    results = fetch_live_assets(api_concepts, max_per_concept=1)
    print(f"Fetched: {len(results)}")
    for item in results:
        sidecar = item.get("sidecar", {})
        print(
            f"- {item.get('concept')} ({item.get('source')}): {sidecar.get('id')} -> {sidecar.get('src')}"
        )


if __name__ == "__main__":
    main()
