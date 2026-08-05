"""
Shared LLM client for the scene-generation pipeline.

Single source of truth for talking to Gemini (via Vertex AI, using Application
Default Credentials — no API key) and to Groq (REST, OpenAI-compatible chat
completions endpoint). Every pipeline stage that needs an LLM call — the prompt
optimizer, the intent extractor, the three scene-architect passes, and the
critic/fixer — imports from here instead of constructing its own client.

This replaces two previously-duplicated Vertex AI client constructors
(`pipeline/scene_architect.py::_make_vertex_client` and
`pipeline/critic_agent.py::_get_gemini_client`), which were identical except
for a `try/except` wrapper. It also replaces `scene_architect.py`'s inline
`_call_architect_groq` REST call with a shared, reusable version.

Note on naming: despite talking to both Gemini and Groq, this module is named
after Gemini since Gemini/Vertex AI is the primary path for every pipeline
stage — Groq is the fallback everywhere it's used.
"""
from __future__ import annotations

import os
from typing import Any

import requests

from core.utils.logger import get_logger

logger = get_logger("gemini_client")

try:
    from google import genai as _genai
    from google.genai.types import HttpOptions as _HttpOptions
    from google.genai import types as _genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

_vertex_client: Any | None = None
_vertex_client_failed = False


def get_vertex_client() -> Any | None:
    """
    Return a cached Vertex AI genai client using ADC (no API key required).

    Cached at module level — constructing the client is cheap but there's no
    reason to rebuild it on every call within a process. If construction
    fails once (e.g. ADC not configured), the failure is cached too so we
    don't retry a doomed client construction on every subsequent call.
    """
    global _vertex_client, _vertex_client_failed
    if _vertex_client is not None:
        return _vertex_client
    if _vertex_client_failed or not GENAI_AVAILABLE:
        return None
    try:
        _vertex_client = _genai.Client(
            vertexai=True,
            project=os.getenv("GCP_PROJECT", "reportevaluator"),
            location=os.getenv("GCP_LOCATION", "us-central1"),
            http_options=_HttpOptions(api_version="v1"),
        )
        return _vertex_client
    except Exception as exc:
        logger.warning("Vertex AI client construction failed: %s", exc)
        _vertex_client_failed = True
        return None


def call_gemini(
    model: str,
    prompt: str,
    system: str,
    *,
    temperature: float = 0.4,
    max_output_tokens: int | None = None,
    thinking_budget: int | None = None,
    json_mode: bool = True,
) -> str | None:
    """
    Single Gemini call via Vertex AI. Returns raw response text, or None on
    any failure (missing SDK, ADC/client construction failure, API error).

    `max_output_tokens`/`thinking_budget` are optional so this one function
    covers both existing call shapes: the architect's plain call (neither
    set) and the critic's bounded-token, thinking-disabled call.
    """
    client = get_vertex_client()
    if client is None:
        return None
    try:
        config_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": temperature,
        }
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
        if max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = max_output_tokens
        if thinking_budget is not None:
            config_kwargs["thinking_config"] = _genai_types.ThinkingConfig(
                thinking_budget=thinking_budget
            )
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=_genai_types.GenerateContentConfig(**config_kwargs),
        )
        return response.text
    except Exception as exc:
        logger.error("Gemini call failed (model=%s): %s", model, exc)
        return None


def call_groq(
    model: str,
    prompt: str,
    system: str,
    *,
    temperature: float = 0.4,
    timeout: int = 60,
) -> str | None:
    """
    Groq REST call (OpenAI-compatible chat completions), JSON object response
    format enforced. Returns raw response text, or None on any failure
    (missing GROQ_API_KEY, network error, malformed response).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    try:
        resp = requests.post(_GROQ_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.error("Groq call failed (model=%s): %s", model, exc)
        return None


def call_llm(
    model_gemini: str,
    model_groq: str,
    prompt: str,
    system: str,
    **kwargs: Any,
) -> tuple[str | None, str | None]:
    """
    Try Gemini first, fall back to Groq on any failure.

    Returns (raw_text, provider) where provider is "gemini", "groq", or None
    if both failed — callers use `provider` as event metadata (e.g. to show a
    "Gemini 2.5 Pro" vs "Groq (fallback)" badge in the pipeline UI).

    `kwargs` are forwarded to `call_gemini` only (temperature/max_output_tokens/
    thinking_budget/json_mode) — Groq fallback always uses call_groq's own
    defaults except temperature, which is shared.
    """
    temperature = kwargs.get("temperature", 0.4)
    raw = call_gemini(model_gemini, prompt, system, **kwargs)
    if raw:
        return raw, "gemini"
    logger.info("Gemini unavailable/failed, falling back to Groq (model=%s)", model_groq)
    raw = call_groq(model_groq, prompt, system, temperature=temperature)
    if raw:
        return raw, "groq"
    return None, None
