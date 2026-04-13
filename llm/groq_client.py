import os
import json
import time
import requests
from pydantic import ValidationError

from llm.scene_schema import SceneSchema
from llm.prompt_templates import build_system_prompt, build_refinement_prompt
from llm.context_manager import ContextManager
from core.utils.logger import get_logger

logger = get_logger("groq_client")
context_manager = ContextManager()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Provide a fallback model and URL
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"
from llm.ollama_client import generate_scene_ollama

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
        if start_idx == -1: return None
        end_idx = text.rfind("}") + 1
        if end_idx <= start_idx: return None
        json_str = text[start_idx:end_idx]
        return json.loads(json_str)
    except Exception:
        return None

def _call_groq(prompt: str) -> str:
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
            {"role": "system", "content": "You are a scene generator. You must output strict, valid JSON. Do not write markdown blocks."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"}
    }
    
    response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=20)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]

def _validate(raw_json: str) -> dict:
    scene = SceneSchema.model_validate_json(raw_json)
    return scene.model_dump()

def generate_scene_groq(command: str, previous_scene: dict | None) -> dict:
    if previous_scene is not None:
        prompt = build_refinement_prompt(previous_scene, command)
    else:
        prompt = f"{build_system_prompt()}\n\nUser command: {command}"
        
    start = time.perf_counter()
    try:
        raw = _call_groq(prompt)
        json_obj = _extract_json_from_text(raw)
        if not json_obj:
            raise ValueError(f"No JSON found in Groq response. Raw response was: {raw[:300]!r}")
        scene = _validate(json.dumps(json_obj))
        elapsed = time.perf_counter() - start
        logger.info("Groq latency: %.1fms", elapsed * 1000)
        return scene

    except ValidationError as first_error:
        logger.warning("Groq validation failed on first attempt: %s", first_error)

        correction_prompt = (
            f"{build_system_prompt()}\n\n"
            f"User command: {command}\n\n"
            f"Your previous response failed validation with this error:\n{first_error}\n\n"
            "Return corrected JSON that includes all required fields for every object."
        )

        try:
            raw = _call_groq(correction_prompt)
            json_obj = _extract_json_from_text(raw)
            if not json_obj:
                raise ValueError(f"No JSON found in Groq correction response. Raw response was: {raw[:300]!r}")

            scene = _validate(json.dumps(json_obj))
            elapsed = time.perf_counter() - start
            logger.info("Groq latency (with retry): %.1fms", elapsed * 1000)
            return scene
        except Exception as second_error:
            logger.error("Groq correction attempt failed: %s", second_error)
            return FALLBACK_SCENE

    except Exception as e:
        logger.error("Groq generation failed: %s", e)
        return FALLBACK_SCENE

def generate_scene(command: str, intent: str = "NEW_SCENE") -> dict:
    from llm.provider_config import get_provider
    provider = get_provider()
    previous_scene = context_manager.last_scene() if intent == "REFINE" else None
    
    if provider == "HYBRID" or provider == "GROQ":
        logger.info("Attempting Groq API...")
        try:
            result = generate_scene_groq(command, previous_scene)
            if result != FALLBACK_SCENE:
                context_manager.add(command, result)
                return result
        except Exception as e:
            logger.warning("Groq failed: %s. Falling back to Ollama...", e)
            
        logger.info("Running Ollama Fallback...")
        raw = generate_scene_ollama(command, previous_scene)
        result = raw if raw is not None else FALLBACK_SCENE
        context_manager.add(command, result)
        return result
    else:
        logger.info("Running Ollama ONLY Mode...")
        raw = generate_scene_ollama(command, previous_scene)
        result = raw if raw is not None else FALLBACK_SCENE
        context_manager.add(command, result)
        return result
