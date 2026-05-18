"""LLM fallback -- last resort, called only for truly unresolvable concepts."""
from __future__ import annotations

import json

SYSTEM_PROMPT = """\
You are a strict 3D object generator. Output ONLY a raw JSON array. No markdown.

Rules (every rule is non-negotiable):
1. Output format: [ { object }, { object }, ... ]
2. Every object needs: id, type, position, scale, material, animation
3. type="primitive" -> must have geometry.type in {sphere,box,cylinder,plane,ring,capsule,torus}
4. type="mesh" -> must have model (string path). Only if you are certain the file exists.
5. material.type always "standard". material.color always "#rrggbb" hex (NOT float array).
6. animation.type in {none,orbit,spin}.
   orbit -> add: "center":[0,0,0], "speed":0.5
   spin  -> add: "axis":[0,1,0], "speed":1.0
7. All numbers finite. position and scale are [x,y,z] arrays. Max 15 objects.
8. Do NOT invent geometry types.
"""


def llm_generate_objects(description: str) -> list[dict] | None:
	prompt = f"Generate 3D objects for: {description}\nOutput only the JSON array."
	raw = _try_groq(prompt)
	return _parse_array(raw) if raw else None


def _try_groq(prompt: str) -> str | None:
	try:
		from llm.groq_client import generate_raw

		return generate_raw(prompt, SYSTEM_PROMPT)
	except Exception:
		return None


def _parse_array(raw: str) -> list[dict] | None:
	try:
		raw = raw.strip()
		if raw.startswith("```"):
			parts = raw.split("```")
			raw = parts[1][4:] if parts[1].startswith("json") else parts[1]
		start, end = raw.find("["), raw.rfind("]") + 1
		if start == -1 or end <= start:
			return None
		return json.loads(raw[start:end])
	except Exception:
		return None
