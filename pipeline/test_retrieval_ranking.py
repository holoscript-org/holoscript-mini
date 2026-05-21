import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.retrieval import retrieve


def main() -> None:
    print("=== Retrieval Ranking Smoke Test ===")
    intent = {
        "objects": ["ring", "tower", "rocketship"],
        "structures": [],
        "systems": [],
        "effects": [],
    }
    results = retrieve(intent)
    assets = results.get("assets", [])
    print(f"Assets: {len(assets)}")
    for item in assets[:10]:
        print(
            f"- {item.get('concept')}: conf={item.get('confidence')} src={item.get('source')} id={item.get('asset_id')}"
        )


if __name__ == "__main__":
    main()
