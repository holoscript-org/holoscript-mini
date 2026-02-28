import json
import os
import time

import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import ValidationError

from llm.scene_schema import SceneSchema
from llm.prompt_templates import build_system_prompt, build_refinement_prompt

load_dotenv()

_API_KEY = os.getenv("GEMINI_API_KEY")
if not _API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")

genai.configure(api_key=_API_KEY)

_MODEL = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config={"response_mime_type": "application/json"},
)

FALLBACK_SCENE = {
    "objects": [
        {
            "id": "fallback_sphere",
            "type": "sphere",
            "position": [0.0, 0.0, 0.0],
            "color": [1.0, 0.84, 0.0],
            "animation": "none",
            "orbit_center": [0.0, 0.0, 0.0],
            "orbit_speed": 0.0,
        }
    ]
}


def _call_gemini(prompt: str) -> str:
    response = _MODEL.generate_content(prompt)
    return response.text


def _validate(raw_json: str) -> dict:
    scene = SceneSchema.model_validate_json(raw_json)
    return scene.model_dump()


def generate_scene(command: str, previous_scene: dict | None) -> dict:
    if previous_scene is not None:
        prompt = build_refinement_prompt(previous_scene, command)
    else:
        prompt = f"{build_system_prompt()}\n\nUser command: {command}"

    start = time.perf_counter()

    try:
        raw = _call_gemini(prompt)
        scene = _validate(raw)
        elapsed = time.perf_counter() - start
        print(f"[gemini_client] latency: {elapsed * 1000:.1f}ms")
        return scene

    except ValidationError as first_error:
        print(f"[gemini_client] ValidationError on first attempt: {first_error}")

        correction_prompt = (
            f"{build_system_prompt()}\n\n"
            f"User command: {command}\n\n"
            f"Your previous response failed validation with this error:\n{first_error}\n\n"
            f"Fix the JSON so it conforms exactly to the schema."
        )

        try:
            raw = _call_gemini(correction_prompt)
            scene = _validate(raw)
            elapsed = time.perf_counter() - start
            print(f"[gemini_client] latency (with retry): {elapsed * 1000:.1f}ms")
            return scene

        except (ValidationError, Exception) as second_error:
            print(f"[gemini_client] Second attempt failed: {second_error}. Returning fallback scene.")
            return FALLBACK_SCENE

    except Exception as e:
        print(f"[gemini_client] Unexpected error: {e}. Returning fallback scene.")
        return FALLBACK_SCENE
