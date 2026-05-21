import json
import os
import time

import requests

from core.utils.logger import get_logger
from llm.context_manager import ContextManager

logger = get_logger("groq_client")
context_manager = ContextManager()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Provide a fallback model and URL
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """
You are a 3D scene generator. Output ONLY raw JSON.
Do not include markdown or code fences.
""".strip()

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

def _extract_json_from_text(text: str) -> dict | None:
    try:
        start_idx = text.find("{")
        if start_idx == -1: return None
        end_idx = text.rfind("}") + 1
        if end_idx <= start_idx: return None
        json_str = text[start_idx:end_idx]
        return json.loads(json_str)
    except Exception:
        return None

def _call_groq(prompt: str, system_prompt: str | None = None) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not found in environment. Please add it to your .env file.")
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    # Tell LLM to output JSON format
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt or "You are a scene generator. You must output strict, valid JSON. Do not write markdown blocks.",
            },
            {"role": "user", "content": prompt}
        ]
    }
    
    response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=20)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]


def generate_raw(prompt: str, system_prompt: str) -> str | None:
    try:
        return _call_groq(prompt, system_prompt=system_prompt)
    except Exception as exc:
        logger.error("Groq request failed: %s", exc)
        return None

def _build_prompt(command: str, previous_scene: dict | None) -> str:
    if previous_scene is not None:
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Current scene:\n{json.dumps(previous_scene, indent=2)}\n\n"
            f"User command: {command}\n\n"
            "Apply the command to the scene. Output ONLY the modified scene as raw JSON."
        )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"User command: {command}\n\n"
        "Output ONLY the JSON object."
    )

def generate_scene_groq(command: str, previous_scene: dict | None) -> dict:
    prompt = _build_prompt(command, previous_scene)
        
    start = time.perf_counter()
    try:
        raw = _call_groq(prompt)
        json_obj = _extract_json_from_text(raw)
        if not json_obj:
            raise ValueError("No JSON found in Groq response.")
        scene = json_obj
        elapsed = time.perf_counter() - start
        logger.info("Groq latency: %.1fms", elapsed * 1000)
        return scene
    except Exception as e:
        logger.error("Groq generation failed: %s", e)
        return FALLBACK_SCENE

def generate_scene(command: str, intent: str = "NEW_SCENE") -> dict:
    previous_scene = context_manager.last_scene() if intent == "REFINE" else None
    logger.info("Attempting Groq API...")
    try:
        result = generate_scene_groq(command, previous_scene)
        if result != FALLBACK_SCENE:
            context_manager.add(command, result)
            return result
    except Exception as e:
        logger.warning("Groq failed: %s. Returning fallback scene...", e)

    context_manager.add(command, FALLBACK_SCENE)
    return FALLBACK_SCENE
