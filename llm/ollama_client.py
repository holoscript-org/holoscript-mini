import os
import json
import time
import requests
from pydantic import ValidationError

from llm.scene_schema import SceneSchema
from llm.prompt_templates import build_system_prompt, build_refinement_prompt

OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:latest")
OLLAMA_TIMEOUT = 300
OLLAMA_WARMUP_TIMEOUT = int(os.getenv("OLLAMA_WARMUP_TIMEOUT", "25"))

FALLBACK_SCENE = {
    "objects": [
        {
            "id": "fallback_sphere",
            "type": "sphere",
            "position": [0.0, 0.0, 0.0],
            "color": [1.0, 0.84, 0.0],
            "secondary_color": [1.0, 0.6, 0.0],
            "size": 1.0,
            "surface_style": "emissive_glow",
            "animation": "none",
            "orbit_center": [0.0, 0.0, 0.0],
            "orbit_speed": 0.0,
        }
    ]
}


def _extract_json_from_text(text: str) -> dict | None:
    try:
        start_idx = text.find("{")
        if start_idx == -1:
            return None
        end_idx = text.rfind("}") + 1
        if end_idx <= start_idx:
            return None
        json_str = text[start_idx:end_idx]
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None


def _call_ollama(prompt: str) -> str | None:
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": "10m",
        }
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")
    except requests.ConnectionError:
        print("[ollama_client] Connection refused: Ollama not running on localhost:11434")
        return None
    except requests.Timeout:
        print(f"[ollama_client] Timeout after {OLLAMA_TIMEOUT}s")
        return None
    except Exception as e:
        print(f"[ollama_client] Error calling Ollama: {e}")
        return None


def warmup_ollama_model() -> bool:
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": "Return {}.",
            "stream": False,
            "format": "json",
            "keep_alive": "10m",
        }
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=OLLAMA_WARMUP_TIMEOUT)
        response.raise_for_status()
        print(f"[ollama_client] Warmed up model: {OLLAMA_MODEL}")
        return True
    except Exception as e:
        print(f"[ollama_client] Warmup failed for {OLLAMA_MODEL}: {e}")
        return False


def _validate(raw_json: str) -> dict:
    scene = SceneSchema.model_validate_json(raw_json)
    return scene.model_dump()


def generate_scene_ollama(command: str, previous_scene: dict | None) -> dict | None:
    if previous_scene is not None:
        prompt = build_refinement_prompt(previous_scene, command)
    else:
        prompt = f"{build_system_prompt()}\n\nUser command: {command}"

    start = time.perf_counter()

    try:
        raw = _call_ollama(prompt)
        if raw is None:
            return None

        json_obj = _extract_json_from_text(raw)
        if json_obj is None:
            print("[ollama_client] Could not extract JSON from response")
            return FALLBACK_SCENE

        raw_json = json.dumps(json_obj)
        scene = _validate(raw_json)
        elapsed = time.perf_counter() - start
        print(f"[ollama_client] latency: {elapsed * 1000:.1f}ms")
        return scene

    except ValidationError as e:
        print(f"[ollama_client] ValidationError: {e}")
        return FALLBACK_SCENE

    except Exception as e:
        print(f"[ollama_client] Unexpected error: {e}. Returning fallback scene.")
        return FALLBACK_SCENE
