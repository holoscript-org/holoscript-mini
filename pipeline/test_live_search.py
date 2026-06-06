import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from pipeline.knowledge_base.mongo_client import lookup_concept
from pipeline.live_search import fetch_live_assets


def _asset_exists(asset_src: str | None) -> bool:
    if not asset_src:
        return False
    return (_ROOT / "core" / asset_src.lstrip("/")).exists()


def main() -> None:
    print("=== Live Search Smoke Test ===")
    concepts = [
        {"concept": "ring", "category": "abstract"},
        {"concept": "tower", "category": "buildings"},
        {"concept": "rocketship", "category": "vehicles"},
    ]
    api_concepts: list[dict[str, str]] = []

    for entry in concepts:
        concept = entry["concept"]
        doc = lookup_concept(concept)
        status = "mongo" if doc and _asset_exists(doc.get("asset_src")) else "api"
        print(f"Concept '{concept}': {status}")
        if status == "api":
            api_concepts.append(entry)

    results = fetch_live_assets(api_concepts, max_per_concept=1)
    print(f"Fetched: {len(results)}")
    for item in results:
        doc = item.get("sidecar", {})
        print(
            f"- {item.get('concept')} ({item.get('source')}): "
            f"{doc.get('_id')} -> {doc.get('asset_src')}"
        )


if __name__ == "__main__":
    main()
