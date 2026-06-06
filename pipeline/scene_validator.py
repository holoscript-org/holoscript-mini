"""
Python port of gui/lib/sceneFactory.ts -> validateScene().

MIRROR STATUS: synchronized with sceneFactory.ts as of 2026-06-06.
If sceneFactory.ts is modified, update this file and the date above.

Corresponding TypeScript lines:
  _is_hex            -> TS line 166-168   isHex()
  _is_vec3           -> TS line 170-172   isVec3()
  _is_num            -> TS line 174-176   isNum()
  _validate_material -> TS line 180-222   validateMaterial()
  _validate_geometry -> TS line 226-282   validateGeometry()
  _validate_animation-> TS line 286-318   validateAnimation()
  _validate_object   -> TS line 322-398   validateObject()
  _validate_parent_refs->TS line 402-432  validateParentRefs()
  validate_scene     -> TS line 436-524   validateScene()
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from core.utils.logger import get_logger

logger = get_logger("scene_validator")

VALID_OBJECT_TYPES  = {"primitive", "mesh"}
VALID_GEOM_TYPES    = {"sphere", "box", "cylinder", "plane", "ring", "capsule", "torus"}
VALID_ANIM_TYPES    = {"none", "orbit", "spin", "physics"}
VALID_PHYSICS_TYPES = {"gravity", "shm", "pendulum", "projectile"}
VALID_LIGHT_TYPES   = {"ambient", "directional", "point", "spot"}

# Physics numeric clamps: (min, max).  Values outside these are clamped silently.
_PHYSICS_CLAMPS: dict[str, tuple[float, float]] = {
    "g":           (0.1,  30.0),
    "restitution": (0.0,   1.0),
    "amplitude":   (0.0,  20.0),
    "frequency":   (0.01, 10.0),
    "damping":     (0.0,   2.0),
    "arm_length":  (0.1,  20.0),
}

DEFAULT_CAMERA = {"position": [0, 5, 20], "target": [0, 0, 0], "fov": 65}
DEFAULT_LIGHTS = [
	{"type": "ambient", "intensity": 0.4},
	{"type": "directional", "intensity": 1.2, "position": [10, 10, 10], "castShadow": True},
]

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

_ROOT = Path(__file__).resolve().parents[1]
_VALID_MESH_EXTS = {".glb", ".gltf"}


def _is_hex(v: Any) -> bool:
	return isinstance(v, str) and bool(_HEX_RE.match(v))


def _is_vec3(v: Any) -> bool:
	return (
		isinstance(v, list)
		and len(v) == 3
		and all(isinstance(n, (int, float)) and math.isfinite(n) for n in v)
	)


def _is_num(v: Any, min_val: float = -math.inf, max_val: float = math.inf) -> bool:
	return isinstance(v, (int, float)) and math.isfinite(v) and min_val <= v <= max_val


def _mesh_exists(model: str) -> bool:
	if not isinstance(model, str) or not model.strip():
		return False
	path = Path(model)
	if path.suffix.lower() not in _VALID_MESH_EXTS:
		return False
	if model.startswith("/"):
		candidate = _ROOT / "core" / model.lstrip("/")
	else:
		candidate = _ROOT / "core" / "assets" / "meshes" / model
	return candidate.exists()


def _validate_material(raw: Any, prefix: str) -> tuple[dict | None, list[str]]:
	errors: list[str] = []

	if not isinstance(raw, dict):
		return None, [f"{prefix}: material is required"]

	if raw.get("type") != "standard":
		errors.append(f'{prefix}.material.type must be "standard", got "{raw.get("type")}"')
	if not _is_hex(raw.get("color")):
		errors.append(
			f'{prefix}.material.color must be "#rrggbb" hex, got "{raw.get("color")}"'
		)
	if not _is_num(raw.get("roughness"), 0, 1):
		errors.append(f"{prefix}.material.roughness must be 0-1, got {raw.get('roughness')}")
	if not _is_num(raw.get("metalness"), 0, 1):
		errors.append(f"{prefix}.material.metalness must be 0-1, got {raw.get('metalness')}")
	if raw.get("opacity") is not None and not _is_num(raw["opacity"], 0, 1):
		errors.append(f"{prefix}.material.opacity must be 0-1")
	if raw.get("transparent") is not None and not isinstance(raw["transparent"], bool):
		errors.append(f"{prefix}.material.transparent must be a boolean")
	if raw.get("emissive") is not None and not _is_hex(raw["emissive"]):
		errors.append(f"{prefix}.material.emissive must be hex")
	if raw.get("emissiveIntensity") is not None and not _is_num(raw["emissiveIntensity"], 0):
		errors.append(f"{prefix}.material.emissiveIntensity must be >= 0")
	for field in ["map", "normalMap", "roughnessMap", "metalnessMap", "emissiveMap"]:
		if raw.get(field) is not None and not isinstance(raw[field], str):
			errors.append(f"{prefix}.material.{field} must be a string path")

	if errors:
		return None, errors

	mat = {
		"type": "standard",
		"color": raw["color"],
		"roughness": raw["roughness"],
		"metalness": raw["metalness"],
	}
	optional = {
		"opacity": raw.get("opacity"),
		"transparent": raw.get("transparent"),
		"map": raw.get("map"),
		"normalMap": raw.get("normalMap"),
		"roughnessMap": raw.get("roughnessMap"),
		"metalnessMap": raw.get("metalnessMap"),
		"emissive": raw.get("emissive"),
		"emissiveMap": raw.get("emissiveMap"),
		"emissiveIntensity": raw.get("emissiveIntensity"),
	}
	mat.update({k: v for k, v in optional.items() if v is not None})
	return mat, []


def _validate_geometry(raw: Any, prefix: str) -> tuple[dict | None, list[str]]:
	if not isinstance(raw, dict):
		return None, [f'{prefix}: geometry is required for type "primitive"']

	t = raw.get("type")
	if t not in VALID_GEOM_TYPES:
		return None, [
			f'{prefix}.geometry.type must be one of {"|".join(sorted(VALID_GEOM_TYPES))}, got "{t}"'
		]

	errors: list[str] = []

	if t in ("sphere", "capsule") and raw.get("radius") is not None:
		if not _is_num(raw["radius"], 0):
			errors.append(f"{prefix}.geometry.radius must be > 0")
	if t == "capsule" and raw.get("length") is not None:
		if not _is_num(raw["length"], 0):
			errors.append(f"{prefix}.geometry.length must be > 0")
	if t == "torus":
		if raw.get("radius") is not None and not _is_num(raw["radius"], 0):
			errors.append(f"{prefix}.geometry.radius must be > 0")
		if raw.get("tube") is not None and not _is_num(raw["tube"], 0):
			errors.append(f"{prefix}.geometry.tube must be > 0")
	if t == "ring":
		if raw.get("innerRadius") is not None and not _is_num(raw["innerRadius"], 0):
			errors.append(f"{prefix}.geometry.innerRadius must be > 0")
		if raw.get("outerRadius") is not None and not _is_num(raw["outerRadius"], 0):
			errors.append(f"{prefix}.geometry.outerRadius must be > 0")
		if (
			_is_num(raw.get("innerRadius"), 0)
			and _is_num(raw.get("outerRadius"), 0)
			and raw["outerRadius"] <= raw["innerRadius"]
		):
			errors.append(f"{prefix}.geometry.outerRadius must be greater than innerRadius")
		if raw.get("thetaSegments") is not None and not _is_num(raw["thetaSegments"], 3):
			errors.append(f"{prefix}.geometry.thetaSegments must be >= 3")
	if t in ("box", "plane") and raw.get("width") is not None:
		if not _is_num(raw["width"], 0):
			errors.append(f"{prefix}.geometry.width must be > 0")
	if raw.get("from") is not None and not _is_vec3(raw["from"]):
		errors.append(f"{prefix}.geometry.from must be [x,y,z]")
	if raw.get("to") is not None and not _is_vec3(raw["to"]):
		errors.append(f"{prefix}.geometry.to must be [x,y,z]")

	if errors:
		return None, errors

	geom = {"type": t}
	optional = {
		"radius": raw.get("radius"),
		"length": raw.get("length"),
		"tube": raw.get("tube"),
		"innerRadius": raw.get("innerRadius"),
		"outerRadius": raw.get("outerRadius"),
		"thetaSegments": raw.get("thetaSegments"),
		"width": raw.get("width"),
		"height": raw.get("height"),
		"depth": raw.get("depth"),
		"from": raw.get("from"),
		"to": raw.get("to"),
	}
	geom.update({k: v for k, v in optional.items() if v is not None})
	return geom, []


def _validate_animation(raw: Any, prefix: str, position: list | None = None) -> tuple[dict | None, list[str]]:
	if raw is None:
		return {"type": "none"}, []
	if not isinstance(raw, dict):
		return None, [f"{prefix}.animation must be an object"]

	t = raw.get("type")
	if t not in VALID_ANIM_TYPES:
		return None, [
			f'{prefix}.animation.type must be one of {"|".join(sorted(VALID_ANIM_TYPES))}, got "{t}"'
		]

	# ── none ─────────────────────────────────────────────────────────────────
	if t == "none":
		return {"type": "none"}, []

	# ── orbit ─────────────────────────────────────────────────────────────────
	if t == "orbit":
		errors: list[str] = []
		if raw.get("center") is not None and not _is_vec3(raw["center"]):
			errors.append(f"{prefix}.animation.center must be [x,y,z]")
		if raw.get("axis") is not None and not _is_vec3(raw["axis"]):
			errors.append(f"{prefix}.animation.axis must be [x,y,z]")
		if raw.get("center_ref") is not None and not isinstance(raw["center_ref"], str):
			errors.append(f"{prefix}.animation.center_ref must be a string")
		if raw.get("speed") is not None and not _is_num(raw["speed"]):
			errors.append(f"{prefix}.animation.speed must be a finite number")
		if raw.get("phase") is not None and not _is_num(raw["phase"]):
			errors.append(f"{prefix}.animation.phase must be a finite number")
		if errors:
			return None, errors
		anim: dict = {"type": "orbit"}
		for k in ("center", "center_ref", "axis", "speed", "phase"):
			if raw.get(k) is not None:
				anim[k] = raw[k]
		return anim, []

	# ── spin ──────────────────────────────────────────────────────────────────
	if t == "spin":
		errors = []
		if raw.get("axis") is not None and not _is_vec3(raw["axis"]):
			errors.append(f"{prefix}.animation.axis must be [x,y,z]")
		if raw.get("speed") is not None and not _is_num(raw["speed"]):
			errors.append(f"{prefix}.animation.speed must be a finite number")
		if errors:
			return None, errors
		anim = {"type": "spin"}
		for k in ("axis", "speed"):
			if raw.get(k) is not None:
				anim[k] = raw[k]
		return anim, []

	# ── physics ───────────────────────────────────────────────────────────────
	if t == "physics":
		physics_type = raw.get("physics_type")
		if physics_type not in VALID_PHYSICS_TYPES:
			return None, [
				f'{prefix}.animation.physics_type must be one of '
				f'{"|".join(sorted(VALID_PHYSICS_TYPES))}, got "{physics_type}"'
			]

		anim = {"type": "physics", "physics_type": physics_type}

		# Clamp all numeric physics params into valid ranges (auto-correct, never reject)
		for field, (lo, hi) in _PHYSICS_CLAMPS.items():
			val = raw.get(field)
			if val is not None and isinstance(val, (int, float)) and math.isfinite(val):
				clamped = max(lo, min(hi, val))
				if clamped != val:
					logger.debug(
						"%s.animation.%s clamped %.4f → %.4f", prefix, field, val, clamped
					)
				anim[field] = clamped

		# floor_y auto-repair: if floor_y >= object's starting y, push it below
		if physics_type in ("gravity", "projectile"):
			floor_y = raw.get("floor_y")
			if floor_y is not None and isinstance(floor_y, (int, float)) and math.isfinite(floor_y):
				start_y = float((position or [0, 0, 0])[1]) if isinstance(position, list) and len(position) >= 2 else 0.0
				if floor_y >= start_y:
					repaired = start_y - 3.0
					logger.debug(
						"%s.animation.floor_y auto-repaired %.2f → %.2f (was at/above start_y %.2f)",
						prefix, floor_y, repaired, start_y,
					)
					floor_y = repaired
				anim["floor_y"] = floor_y

		# Pivot point (pendulum)
		pivot = raw.get("pivot")
		if pivot is not None and _is_vec3(pivot):
			anim["pivot"] = pivot

		# Initial velocity (projectile)
		iv = raw.get("initial_velocity")
		if iv is not None and _is_vec3(iv):
			anim["initial_velocity"] = iv

		# SHM oscillation axis — string "x"|"y"|"z"
		if physics_type == "shm":
			axis = raw.get("axis")
			if axis in ("x", "y", "z"):
				anim["axis"] = axis

		return anim, []

	return {"type": "none"}, []


def _validate_object(raw: Any, index: int) -> tuple[dict | None, list[str]]:
	errors: list[str] = []

	if not isinstance(raw, dict):
		return None, [f"objects[{index}]: must be an object"]

	prefix = f'objects[{index}](id="{raw.get("id")}")'

	if not isinstance(raw.get("id"), str) or not raw["id"].strip():
		errors.append(f"{prefix}: id must be a non-empty string")

	obj_type = raw.get("type")
	if obj_type not in VALID_OBJECT_TYPES:
		return None, [f'{prefix}: type must be "primitive" or "mesh", got "{obj_type}"']

	if raw.get("parent") is not None:
		if not isinstance(raw["parent"], str) or not raw["parent"].strip():
			errors.append(f"{prefix}: parent must be a non-empty string id")

	if not _is_vec3(raw.get("position")):
		errors.append(f"{prefix}: position must be [x,y,z] of finite numbers")

	geom = None
	if obj_type == "primitive":
		geom, ge = _validate_geometry(raw.get("geometry"), prefix)
		errors.extend(ge)
	else:
		if not isinstance(raw.get("model"), str) or not raw["model"].strip():
			errors.append(
				f'{prefix}: model (path to .glb/.gltf) is required for type "mesh"'
			)
		elif not _mesh_exists(raw["model"]):
			errors.append(f"{prefix}: model path not found on disk: {raw['model']}")

	mat, me = _validate_material(raw.get("material"), prefix)
	errors.extend(me)

	if raw.get("rotation") is not None and not _is_vec3(raw["rotation"]):
		errors.append(f"{prefix}: rotation must be [rx,ry,rz]")
	if raw.get("scale") is not None and not _is_vec3(raw["scale"]):
		errors.append(f"{prefix}: scale must be [sx,sy,sz]")
	if raw.get("label") is not None and not isinstance(raw["label"], str):
		errors.append(f"{prefix}: label must be a string")

	anim, ae = _validate_animation(raw.get("animation"), prefix, position=raw.get("position"))
	errors.extend(ae)

	position_ok = _is_vec3(raw.get("position"))
	type_ok = (
		(obj_type == "primitive" and geom is not None)
		or (obj_type == "mesh" and isinstance(raw.get("model"), str) and raw["model"].strip())
	)
	if not position_ok or mat is None or not type_ok:
		return None, errors

	obj = {
		"id": raw["id"],
		"type": obj_type,
		"position": raw["position"],
		"material": mat,
		"animation": anim or {"type": "none"},
	}
	optional = {
		"parent": raw["parent"]
		if isinstance(raw.get("parent"), str) and raw["parent"].strip()
		else None,
		"geometry": geom if obj_type == "primitive" else None,
		"model": raw.get("model") if obj_type == "mesh" else None,
		"rotation": raw["rotation"] if _is_vec3(raw.get("rotation")) else None,
		"scale": raw["scale"] if _is_vec3(raw.get("scale")) else None,
		"label": raw["label"] if isinstance(raw.get("label"), str) else None,
	}
	obj.update({k: v for k, v in optional.items() if v is not None})
	return obj, errors


def _validate_parent_refs(objects: list[dict]) -> list[str]:
	errors: list[str] = []
	id_set = {o["id"] for o in objects}

	for obj in objects:
		parent = obj.get("parent")
		if parent is None:
			continue
		if parent not in id_set:
			errors.append(
				f'objects(id="{obj["id"]}"): parent "{parent}" references an unknown id'
			)
			continue
		if parent == obj["id"]:
			errors.append(f'objects(id="{obj["id"]}"): parent cannot reference self')
			continue
		visited: set[str] = set()
		cur: str | None = obj["id"]
		parent_map = {o["id"]: o.get("parent") for o in objects}
		while cur is not None:
			if cur in visited:
				errors.append(
					f'objects(id="{obj["id"]}"): circular parent dependency detected'
				)
				break
			visited.add(cur)
			cur = parent_map.get(cur)

	return errors


def validate_scene(raw: Any) -> dict:
	all_errors: list[str] = []

	if not isinstance(raw, dict):
		return {
			"scene": {"objects": [], "lights": DEFAULT_LIGHTS, "camera": DEFAULT_CAMERA},
			"errors": [],
			"fatal": "Scene root must be a JSON object",
		}

	valid_objects: list[dict] = []
	if not isinstance(raw.get("objects"), list) or len(raw["objects"]) == 0:
		all_errors.append("scene.objects must be a non-empty array")
	else:
		for i, obj_raw in enumerate(raw["objects"]):
			obj, errs = _validate_object(obj_raw, i)
			all_errors.extend(errs)
			if obj:
				valid_objects.append(obj)

	all_errors.extend(_validate_parent_refs(valid_objects))

	lights = DEFAULT_LIGHTS
	if raw.get("lights") is not None:
		if not isinstance(raw["lights"], list):
			all_errors.append("scene.lights must be an array")
		else:
			valid_lights = []
			for light in raw["lights"]:
				if not isinstance(light, dict):
					continue
				if light.get("type") not in VALID_LIGHT_TYPES:
					all_errors.append(
						f'light: type must be one of {"|".join(sorted(VALID_LIGHT_TYPES))}, got "{light.get("type")}"'
					)
					continue
				if not _is_num(light.get("intensity"), 0):
					all_errors.append(f"light({light['type']}): intensity must be >= 0")
					continue
				if light.get("color") is not None and not _is_hex(light["color"]):
					all_errors.append(f"light({light['type']}): color must be \"#rrggbb\" hex")
					continue
				if light.get("position") is not None and not _is_vec3(light["position"]):
					all_errors.append(f"light({light['type']}): position must be [x,y,z]")
					continue
				light_obj = {
					"type": light["type"],
					"intensity": light["intensity"],
					"color": light.get("color", "#ffffff"),
				}
				optional = {
					"position": light.get("position"),
					"castShadow": light.get("castShadow", False),
				}
				light_obj.update({k: v for k, v in optional.items() if v is not None})
				valid_lights.append(light_obj)
			if valid_lights:
				lights = valid_lights

	camera = DEFAULT_CAMERA
	if raw.get("camera") is not None:
		cam = raw["camera"]
		if not isinstance(cam, dict):
			all_errors.append("scene.camera must be an object")
		elif not _is_vec3(cam.get("position")):
			all_errors.append("scene.camera.position must be [x,y,z]")
		elif not _is_vec3(cam.get("target")):
			all_errors.append("scene.camera.target must be [x,y,z]")
		else:
			camera = {
				"position": cam["position"],
				"target": cam["target"],
				"fov": cam["fov"] if _is_num(cam.get("fov"), 1, 179) else DEFAULT_CAMERA["fov"],
			}

	return {
		"scene": {
			"name": raw.get("name") if isinstance(raw.get("name"), str) else None,
			"objects": valid_objects,
			"lights": lights,
			"camera": camera,
		},
		"errors": all_errors,
		"fatal": None,
	}


def is_valid(result: dict) -> bool:
	return result.get("fatal") is None and len(result.get("scene", {}).get("objects", [])) > 0
