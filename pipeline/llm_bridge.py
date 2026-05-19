"""LLM semantic assistant for unresolved concepts.

This module must not generate scene objects or mesh paths. It only provides
search/refinement hints to help retrieval and live search.
"""
from __future__ import annotations

import json

SYSTEM_PROMPT = """\
You are a semantic assistant for 3D scene retrieval.
Output ONLY a JSON object with search hints. No markdown.

Rules (every rule is non-negotiable):
1. Output format: {"suggested_search_terms": [..], "semantic_category": "..."}
2. suggested_search_terms must be a list of short strings (1-5 words each).
3. semantic_category should be a single lowercase token (e.g., anatomy, medical, abstract).
4. Do NOT generate objects, meshes, or model paths.
5. Do NOT output scene JSON.
"""


def llm_suggest_search_terms(text: str) -> dict | None:
	prompt = (
		"Given this unresolved concept or phrase, suggest search terms and a semantic category.\n"
		f"Input: {text}\n"
		"Output only the JSON object."
	)
	raw = _try_groq(prompt)
	return _parse_object(raw) if raw else None


def _try_groq(prompt: str) -> str | None:
	try:
		from llm.groq_client import generate_raw

		return generate_raw(prompt, SYSTEM_PROMPT)
	except Exception:
		return None


def _parse_object(raw: str) -> dict | None:
	try:
		raw = raw.strip()
		if raw.startswith("```"):
			parts = raw.split("```")
			raw = parts[1][4:] if parts[1].startswith("json") else parts[1]
		start, end = raw.find("{"), raw.rfind("}") + 1
		if start == -1 or end <= start:
			return None
		parsed = json.loads(raw[start:end])
		return parsed if isinstance(parsed, dict) else None
	except Exception:
		return None
