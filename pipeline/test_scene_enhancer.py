import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from pipeline.scene_enhancer import enhance_scene


def main() -> None:
    base_scene = {
        "name": "Base Test",
        "objects": [
            {
                "id": "ring",
                "type": "primitive",
                "geometry": {"type": "torus", "radius": 1.2, "tube": 0.3},
                "position": [0, 0, 0],
                "scale": [1, 1, 1],
                "material": {
                    "type": "standard",
                    "color": "#22c55e",
                    "roughness": 0.4,
                    "metalness": 0.2,
                },
                "animation": {"type": "none"},
            }
        ],
        "lights": [
            {"type": "ambient", "intensity": 0.4},
            {"type": "directional", "intensity": 1.2, "position": [10, 10, 10], "castShadow": True},
        ],
        "camera": {"position": [0, 5, 12], "target": [0, 0, 0], "fov": 65},
    }
    intent = {"objects": ["ring"], "structures": [], "systems": [], "effects": []}
    components = {"assets": [], "generators": [], "effects": []}

    result = enhance_scene(
        "a glowing ring in a cinematic space",
        intent,
        components,
        base_scene,
    )

    print("=== Scene Enhancer Smoke Test ===")
    print(result)


if __name__ == "__main__":
    main()
