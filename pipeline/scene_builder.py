"""Assembles scene dict from retrieved components."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from pipeline import effects, generators

_KB = Path(__file__).parent / "knowledge_base"
with (_KB / "concept_map.json").open(encoding="utf-8") as f:
	_CONCEPT_MAP = json.load(f)

DEFAULT_LIGHTS = [
	{"type": "ambient", "intensity": 0.4, "color": "#ffffff"},
	{
		"type": "directional",
		"intensity": 1.2,
		"color": "#ffffff",
		"position": [10, 10, 10],
		"castShadow": True,
	},
]


def _uid(base: str, used: set[str]) -> str:
	cand, n = base, 0
	while cand in used:
		n += 1
		cand = f"{base}_{n}"
	return cand


def _camera(objects: list[dict]) -> dict:
	if not objects:
		return {"position": [0, 5, 20], "target": [0, 0, 0], "fov": 65}
	xs = [o["position"][0] for o in objects]
	ys = [o["position"][1] for o in objects]
	zs = [o["position"][2] for o in objects]
	cx = (max(xs) + min(xs)) / 2
	cy = (max(ys) + min(ys)) / 2
	cz = (max(zs) + min(zs)) / 2
	spread = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 10)
	return {
		"position": [round(cx, 2), round(cy + spread * 0.5, 2), round(cz + spread * 1.8, 2)],
		"target": [round(cx, 2), round(cy, 2), round(cz, 2)],
		"fov": 65,
	}


def _name(intent: dict) -> str:
	parts = intent.get("objects", []) + intent.get("structures", []) + intent.get("systems", [])
	return " + ".join(p.replace("_", " ").title() for p in parts[:3]) or "Generated Scene"


def build_scene(components: dict, intent: dict, seed: int = 42) -> dict:
	rng = random.Random(seed)
	all_objs: list[dict] = []
	used: set[str] = set()

	solo = [a for a in components.get("assets", []) if not a.get("is_generator")]
	n = max(len(solo), 1)
	for i, item in enumerate(solo):
		concept = item["concept"]
		obj_id = _uid(concept.lower().replace(" ", "_"), used)
		used.add(obj_id)
		angle = i * 2 * math.pi / n
		radius = rng.uniform(4, 8)
		pos = [round(radius * math.cos(angle), 2), 0.0, round(radius * math.sin(angle), 2)]
		sv = rng.uniform(0.85, 1.15)
		obj = generators.make_object(
			concept,
			item.get("asset_src"),
			item.get("sidecar"),
			item.get("category", "abstract"),
			pos,
			[round(sv, 3)] * 3,
			obj_id,
			concept.replace("_", " ").title(),
		)
		obj["concept_id"] = item.get("concept_id", concept)
		obj["semantic_locked"] = True
		all_objs.append(obj)

	planet = next(
		(
			o
			for o in all_objs
			if any(tag in o["id"] for tag in ["planet", "mars", "earth", "moon", "world", "saturn", "jupiter"])
		),
		None,
	)
	scene_center = list(planet["position"]) if planet else [0, 0, 0]

	for item in components.get("assets", []):
		if not item.get("is_generator"):
			continue
		concept = item["concept"]
		gen_type = item["generator_type"]
		count = rng.randint(*item["generator_count"])
		base_id = _uid(concept.lower(), used)
		if gen_type == "scatter":
			objs = generators.scatter(
				item.get("asset_src"),
				item.get("sidecar"),
				item.get("category", "abstract"),
				count,
				item["generator_radius"],
				scene_center,
				base_id,
				scene_center[1],
				rng.randint(0, 9999),
			)
		elif gen_type == "grid":
			objs = generators.grid(
				item.get("asset_src"),
				item.get("sidecar"),
				item.get("category", "abstract"),
				count,
				item["generator_spacing"],
				scene_center,
				base_id,
				scene_center[1],
				rng.randint(0, 9999),
			)
		else:
			objs = []

		for obj in objs:
			obj["id"] = _uid(obj["id"], used)
			used.add(obj["id"])
			obj["concept_id"] = item.get("concept_id", concept)
			obj["semantic_locked"] = True
		all_objs.extend(objs)

	for spec in components.get("generators", []):
		count = rng.randint(*spec["count"])
		central = spec.get("central_sidecar")
		satellite = spec.get("satellite_sidecar")
		objs = generators.orbit_cluster(
			(central or {}).get("src"),
			central,
			(satellite or {}).get("src"),
			satellite,
			count,
			(6, 18),
			_uid("center", used),
			spec["concept"],
			(central or {}).get("category", "abstract"),
			(satellite or {}).get("category", "abstract"),
			rng.randint(0, 9999),
		)
		for obj in objs:
			obj["id"] = _uid(obj["id"], used)
			used.add(obj["id"])
			obj["concept_id"] = spec.get("concept_id", spec.get("concept"))
			obj["semantic_locked"] = True
		all_objs.extend(objs)

	for eff in components.get("effects", []):
		all_objs = effects.apply_effect(all_objs, eff["handler"])

	for obj in all_objs:
		if "animation" not in obj:
			obj["animation"] = {"type": "none"}

	return {
		"name": _name(intent),
		"objects": all_objs,
		"lights": DEFAULT_LIGHTS,
		"camera": _camera(all_objs),
	}
