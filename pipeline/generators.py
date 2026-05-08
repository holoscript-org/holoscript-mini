"""
Deterministic generators. All use seeded random -- same input = same output.
No hardcoded colors or geometry values. Appearance derived from asset sidecar.
"""
from __future__ import annotations

import math
import random

_CATEGORY_FALLBACK = {
	"humans": {"geom": "capsule", "params": {"radius": 0.3, "length": 1.0}},
	"vehicles": {"geom": "box", "params": {"width": 2.0, "height": 0.8, "depth": 3.5}},
	"buildings": {"geom": "box", "params": {"width": 1.2, "height": 3.0, "depth": 1.2}},
	"trees": {"geom": "cylinder", "params": {"from": [0, 0, 0], "to": [0, 2.5, 0]}},
	"planets": {"geom": "sphere", "params": {"radius": 1.5}},
	"satellites": {"geom": "box", "params": {"width": 0.8, "height": 0.4, "depth": 0.8}},
	"abstract": {"geom": "torus", "params": {"radius": 1.0, "tube": 0.3}},
}

_NEUTRAL_MATERIAL = {
	"type": "standard",
	"color": "#888888",
	"roughness": 0.6,
	"metalness": 0.1,
}


def make_object(
	concept: str,
	asset_src: str | None,
	sidecar: dict | None,
	category: str,
	position: list,
	scale: list,
	obj_id: str,
	label: str | None = None,
	animation: dict | None = None,
) -> dict:
	mat = dict(_NEUTRAL_MATERIAL)
	if asset_src:
		return {
			"id": obj_id,
			"type": "mesh",
			"model": asset_src,
			"position": position,
			"scale": scale,
			"material": mat,
			**({"label": label} if label else {}),
			"animation": animation or {"type": "none"},
		}
	fb = _CATEGORY_FALLBACK.get(category, {"geom": "sphere", "params": {"radius": 1.0}})
	return {
		"id": obj_id,
		"type": "primitive",
		"geometry": {"type": fb["geom"], **fb["params"]},
		"position": position,
		"scale": scale,
		"material": mat,
		**({"label": label} if label else {}),
		"animation": animation or {"type": "none"},
	}


def scatter(
	asset_src,
	sidecar,
	category,
	count,
	radius,
	center,
	base_id,
	y_base=0.0,
	seed=42,
):
	rng, objs = random.Random(seed), []
	for i in range(count):
		angle = rng.uniform(0, 2 * math.pi)
		dist = rng.uniform(radius * 0.5, radius)
		sv = rng.uniform(0.8, 1.3)
		obj = make_object(
			category,
			asset_src,
			sidecar,
			category,
			[round(dist * math.cos(angle), 3), round(y_base, 3), round(dist * math.sin(angle), 3)],
			[round(sv, 3)] * 3,
			f"{base_id}_{i}",
		)
		obj["rotation"] = [0, round(rng.uniform(0, 360), 1), 0]
		objs.append(obj)
	return objs


def grid(
	asset_src,
	sidecar,
	category,
	count,
	spacing,
	center,
	base_id,
	y_base=0.0,
	seed=42,
):
	rng = random.Random(seed)
	cols = math.ceil(math.sqrt(count))
	rows = math.ceil(count / cols)
	objs, idx = [], 0
	for row in range(rows):
		for col in range(cols):
			if idx >= count:
				break
			x = center[0] + (col - cols / 2.0) * spacing
			z = center[2] + (row - rows / 2.0) * spacing
			sv = rng.uniform(0.8, 1.4)
			obj = make_object(
				category,
				asset_src,
				sidecar,
				category,
				[round(x, 3), round(y_base, 3), round(z, 3)],
				[round(sv, 3)] * 3,
				f"{base_id}_{idx}",
			)
			obj["rotation"] = [0, round(rng.uniform(0, 360), 1), 0]
			objs.append(obj)
			idx += 1
	return objs


def orbit_cluster(
	central_src,
	central_sid,
	satellite_src,
	satellite_sid,
	count,
	orbit_radii,
	central_id,
	base_id,
	central_cat="abstract",
	satellite_cat="abstract",
	seed=42,
):
	rng = random.Random(seed)
	central = make_object(
		central_id,
		central_src,
		central_sid,
		central_cat,
		[0, 0, 0],
		[1, 1, 1],
		central_id,
		central_id.replace("_", " ").title(),
	)
	sats = []
	for i in range(count):
		radius = rng.uniform(*orbit_radii)
		phase = rng.uniform(0, 2 * math.pi)
		sv = rng.uniform(0.4, 1.0)
		anim = {
			"type": "orbit",
			"center": [0, 0, 0],
			"speed": round(rng.uniform(0.2, 1.2), 3),
			"phase": phase,
		}
		sats.append(
			make_object(
				f"{base_id}_sat",
				satellite_src,
				satellite_sid,
				satellite_cat,
				[round(radius * math.cos(phase), 3), 0, round(radius * math.sin(phase), 3)],
				[round(sv, 3)] * 3,
				f"{base_id}_{i}",
				None,
				anim,
			)
		)
	return [central] + sats
