"""Structural repairs without LLM. Max 3 iterations."""
from __future__ import annotations

DEMO_FALLBACK = {
	"name": "Fallback",
	"objects": [
		{
			"id": "fallback",
			"type": "primitive",
			"geometry": {"type": "sphere", "radius": 2.0},
			"position": [0, 0, 0],
			"scale": [1, 1, 1],
			"material": {
				"type": "standard",
				"color": "#ffcc22",
				"roughness": 0.3,
				"metalness": 0.0,
			},
			"animation": {"type": "spin", "axis": [0, 1, 0], "speed": 0.5},
		}
	],
	"lights": [
		{"type": "ambient", "intensity": 0.4},
		{"type": "directional", "intensity": 1.2, "position": [10, 10, 10], "castShadow": True},
	],
	"camera": {"position": [0, 5, 15], "target": [0, 0, 0], "fov": 65},
}


def _fix_colors(scene: dict) -> dict:
	for obj in scene.get("objects", []):
		mat = obj.get("material", {})
		col = mat.get("color")
		if isinstance(col, list) and len(col) == 3:
			r, g, b = [max(0, min(255, int(x * 255))) for x in col]
			mat["color"] = f"#{r:02x}{g:02x}{b:02x}"
		elif not isinstance(col, str) or not col.startswith("#"):
			mat["color"] = "#888888"
	return scene


def _fix_material_types(scene: dict) -> dict:
	for obj in scene.get("objects", []):
		mat = obj.get("material", {})
		if mat.get("type") != "standard":
			mat["type"] = "standard"
		if not _is_num_01(mat.get("roughness")):
			mat["roughness"] = 0.6
		if not _is_num_01(mat.get("metalness")):
			mat["metalness"] = 0.1
	return scene


def _is_num_01(value) -> bool:
	return isinstance(value, (int, float)) and 0 <= value <= 1


def _fix_animations(scene: dict) -> dict:
	valid = {"none", "orbit", "spin", "physics"}
	for obj in scene.get("objects", []):
		anim = obj.get("animation")
		if not isinstance(anim, dict) or anim.get("type") not in valid:
			obj["animation"] = {"type": "none"}
	return scene


def _fix_missing_meshes(scene: dict, bad_paths: set[str] | None = None) -> dict:
	"""Convert only the specific broken mesh objects to sphere primitives.
	If bad_paths is None, converts all mesh objects (legacy fallback)."""
	from pathlib import Path as _Path
	_root = _Path(__file__).resolve().parents[1]

	for i, obj in enumerate(scene.get("objects", [])):
		if obj.get("type") != "mesh":
			continue
		model = obj.get("model", "")
		if bad_paths is not None and model not in bad_paths:
			continue
		# If bad_paths is None (no info), check disk ourselves
		if bad_paths is None:
			candidate = _root / "core" / model.lstrip("/")
			if candidate.exists():
				continue
		scene["objects"][i] = {
			"id":       obj.get("id", f"fallback_{i}"),
			"type":     "primitive",
			"geometry": {"type": "sphere", "radius": 1.0},
			"position": obj.get("position", [0, 0, 0]),
			"scale":    obj.get("scale", [1, 1, 1]),
			"material": obj.get("material") or {
				"type": "standard", "color": "#888888",
				"roughness": 0.6, "metalness": 0.1,
			},
			"animation": obj.get("animation") or {"type": "none"},
			**({"label":  obj["label"]}  if obj.get("label")  else {}),
			**({"parent": obj["parent"]} if obj.get("parent") else {}),
		}
	return scene


def _fix_geometries(scene: dict) -> dict:
	valid = {"sphere", "box", "cylinder", "plane", "ring", "capsule", "torus"}
	for obj in scene.get("objects", []):
		if obj.get("type") == "primitive":
			geom = obj.get("geometry", {})
			if geom.get("type") not in valid:
				obj["geometry"] = {"type": "sphere", "radius": 1.0}
	return scene


def _fix_transparency(scene: dict) -> dict:
	for obj in scene.get("objects", []):
		mat = obj.get("material", {})
		op = mat.get("opacity")
		if isinstance(op, (int, float)) and op < 1.0:
			mat["transparent"] = True
	return scene


def _fix_required(scene: dict) -> dict:
	for obj in scene.get("objects", []):
		if not isinstance(obj.get("position"), list) or len(obj.get("position", [])) != 3:
			obj["position"] = [0, 0, 0]
		if not isinstance(obj.get("scale"), list):
			obj["scale"] = [1, 1, 1]
		if "material" not in obj:
			obj["material"] = {
				"type": "standard",
				"color": "#888888",
				"roughness": 0.6,
				"metalness": 0.1,
			}
		if "animation" not in obj:
			obj["animation"] = {"type": "none"}
	return scene


def repair(scene: dict, errors: list[str], max_iterations: int = 3) -> dict:
	current = scene
	for _ in range(max_iterations):
		joined = " ".join(errors).lower()
		if "color" in joined:
			current = _fix_colors(current)
		if "material.type" in joined:
			current = _fix_material_types(current)
		if "animation.type" in joined:
			current = _fix_animations(current)
		if "geometry.type" in joined:
			current = _fix_geometries(current)
		if "transparent" in joined:
			current = _fix_transparency(current)
		if "model path not found" in joined:
			import re as _re
			bad = set(_re.findall(r"model path not found[^:]*:\s*(\S+)", joined))
			current = _fix_missing_meshes(current, bad_paths=bad if bad else None)
		current = _fix_required(current)
		if current.get("objects"):
			return current
	return DEMO_FALLBACK
