"""
pipeline/scene_validator.py — the Python half of the scene schema contract.

This validator is a hand-maintained mirror of gui/lib/sceneFactory.ts. Both
rebuild objects field-by-field and DROP unknown keys, so a field added to only
one side vanishes silently at the other boundary — which is exactly what
happened when `summary`/`description` were introduced. See test_schema_mirror.py
for the cross-language drift guard.
"""
from __future__ import annotations

import copy

import pytest

from pipeline.scene_validator import (
    DEFAULT_CAMERA,
    DEFAULT_LIGHTS,
    VALID_ANIM_TYPES,
    VALID_GEOM_TYPES,
    VALID_LIGHT_TYPES,
    VALID_OBJECT_TYPES,
    VALID_PHYSICS_TYPES,
    is_valid,
    validate_scene,
)


# ── happy path ───────────────────────────────────────────────────────────────

def test_minimal_scene_validates_clean(minimal_scene):
    result = validate_scene(minimal_scene)
    assert result["fatal"] is None
    assert result["errors"] == []
    assert is_valid(result)
    assert len(result["scene"]["objects"]) == 1


def test_scene_name_preserved(minimal_scene):
    assert validate_scene(minimal_scene)["scene"]["name"] == "Test Scene"


# ── educational fields (added 2026-08-05) ────────────────────────────────────

def test_summary_survives_validation(minimal_scene):
    """Regression: unknown root keys are dropped, so `summary` needed explicit
    passthrough. Losing it silently blanks the Scene Summary panel."""
    minimal_scene["summary"] = "Demonstrates gravity and inelastic collision."
    out = validate_scene(minimal_scene)["scene"]
    assert out["summary"] == "Demonstrates gravity and inelastic collision."


def test_description_survives_validation(minimal_scene):
    minimal_scene["objects"][0]["description"] = "Falls under gravity."
    obj = validate_scene(minimal_scene)["scene"]["objects"][0]
    assert obj["description"] == "Falls under gravity."


def test_non_string_summary_becomes_none_without_error(minimal_scene):
    minimal_scene["summary"] = 123
    out = validate_scene(minimal_scene)
    assert out["scene"]["summary"] is None
    assert out["fatal"] is None


def test_non_string_description_is_an_error(minimal_scene):
    minimal_scene["objects"][0]["description"] = 99
    errors = validate_scene(minimal_scene)["errors"]
    assert any("description must be a string" in e for e in errors)


def test_object_without_description_omits_the_key(minimal_scene):
    obj = validate_scene(minimal_scene)["scene"]["objects"][0]
    assert obj.get("description") is None


# ── root-level failures ──────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [None, [], "scene", 42])
def test_non_dict_root_is_fatal(bad):
    result = validate_scene(bad)
    assert result["fatal"] is not None
    assert not is_valid(result)


def test_empty_objects_is_an_error(minimal_scene):
    minimal_scene["objects"] = []
    result = validate_scene(minimal_scene)
    assert any("non-empty array" in e for e in result["errors"])
    assert not is_valid(result)


# ── object validation ────────────────────────────────────────────────────────

def test_bad_object_dropped_but_scene_survives(minimal_scene, minimal_object):
    bad = copy.deepcopy(minimal_object)
    bad["id"] = "broken"
    bad["position"] = "not a vector"
    minimal_scene["objects"].append(bad)
    result = validate_scene(minimal_scene)
    ids = [o["id"] for o in result["scene"]["objects"]]
    assert ids == ["obj_a"]          # good object kept
    assert result["errors"]           # bad one reported


@pytest.mark.parametrize("bad_type", ["sphere", "Primitive", "", None])
def test_invalid_object_type_rejected(minimal_scene, bad_type):
    minimal_scene["objects"][0]["type"] = bad_type
    result = validate_scene(minimal_scene)
    assert result["scene"]["objects"] == []


def test_mesh_requires_model_path(minimal_scene):
    obj = minimal_scene["objects"][0]
    obj["type"] = "mesh"
    obj.pop("geometry")
    result = validate_scene(minimal_scene)
    assert any("model" in e for e in result["errors"])


def test_mesh_with_nonexistent_file_rejected(minimal_scene):
    obj = minimal_scene["objects"][0]
    obj["type"] = "mesh"
    obj.pop("geometry")
    obj["model"] = "/assets/meshes/abstract/definitely_not_here.glb"
    result = validate_scene(minimal_scene)
    assert any("not found on disk" in e for e in result["errors"])


@pytest.mark.parametrize("field", ["roughness", "metalness"])
def test_material_ranges_enforced(minimal_scene, field):
    minimal_scene["objects"][0]["material"][field] = 5.0
    result = validate_scene(minimal_scene)
    assert any(field in e for e in result["errors"])


@pytest.mark.parametrize("color", ["red", "#fff", "#GGGGGG", 123, None])
def test_material_color_must_be_hex(minimal_scene, color):
    minimal_scene["objects"][0]["material"]["color"] = color
    result = validate_scene(minimal_scene)
    assert any("color" in e for e in result["errors"])


# ── geometry ─────────────────────────────────────────────────────────────────

def test_ring_outer_must_exceed_inner(minimal_scene):
    minimal_scene["objects"][0]["geometry"] = {
        "type": "ring", "innerRadius": 2.0, "outerRadius": 1.0, "thetaSegments": 64,
    }
    result = validate_scene(minimal_scene)
    assert any("outerRadius" in e for e in result["errors"])


def test_unknown_geometry_type_rejected(minimal_scene):
    minimal_scene["objects"][0]["geometry"] = {"type": "dodecahedron"}
    result = validate_scene(minimal_scene)
    assert result["scene"]["objects"] == []


# ── animation & physics ──────────────────────────────────────────────────────

def test_missing_animation_defaults_to_none(minimal_scene):
    minimal_scene["objects"][0].pop("animation")
    obj = validate_scene(minimal_scene)["scene"]["objects"][0]
    assert obj["animation"] == {"type": "none"}


def test_physics_params_are_clamped_not_rejected(minimal_scene):
    """Out-of-range physics values must auto-correct — never fail the scene."""
    minimal_scene["objects"][0]["animation"] = {
        "type": "physics", "physics_type": "gravity",
        "g": 999.0, "restitution": 5.0, "floor_y": -5.0,
    }
    result = validate_scene(minimal_scene)
    anim = result["scene"]["objects"][0]["animation"]
    assert anim["g"] == 30.0           # clamped to max
    assert anim["restitution"] == 1.0  # clamped to max
    assert result["fatal"] is None


def test_floor_y_above_start_is_auto_repaired(minimal_scene):
    """floor_y at/above the object's y means it never falls — must be pushed down."""
    obj = minimal_scene["objects"][0]
    obj["position"] = [0, 2, 0]
    obj["animation"] = {
        "type": "physics", "physics_type": "gravity", "g": 9.8, "floor_y": 5.0,
    }
    anim = validate_scene(minimal_scene)["scene"]["objects"][0]["animation"]
    assert anim["floor_y"] < 2


def test_invalid_physics_type_degrades_to_none_but_keeps_object(minimal_scene):
    """
    A bad animation must not discard an otherwise-valid object — only
    position/material/type failures do that. The animation degrades to
    {"type": "none"} and the problem is reported in errors.
    """
    minimal_scene["objects"][0]["animation"] = {
        "type": "physics", "physics_type": "teleport",
    }
    result = validate_scene(minimal_scene)
    objects = result["scene"]["objects"]
    assert len(objects) == 1
    assert objects[0]["animation"] == {"type": "none"}
    assert any("physics_type" in e for e in result["errors"])


# ── parent references ────────────────────────────────────────────────────────

def test_unknown_parent_reported(minimal_scene, minimal_object):
    child = copy.deepcopy(minimal_object)
    child["id"] = "child"
    child["parent"] = "ghost"
    minimal_scene["objects"].append(child)
    result = validate_scene(minimal_scene)
    assert any("unknown id" in e for e in result["errors"])


def test_self_parent_reported(minimal_scene):
    minimal_scene["objects"][0]["parent"] = "obj_a"
    result = validate_scene(minimal_scene)
    assert any("cannot reference self" in e for e in result["errors"])


def test_circular_parent_chain_detected(minimal_scene, minimal_object):
    a = minimal_scene["objects"][0]
    a["parent"] = "obj_b"
    b = copy.deepcopy(minimal_object)
    b["id"] = "obj_b"
    b["parent"] = "obj_a"
    minimal_scene["objects"].append(b)
    result = validate_scene(minimal_scene)
    assert any("circular" in e for e in result["errors"])


# ── lights & camera fallbacks ────────────────────────────────────────────────

def test_missing_lights_fall_back_to_defaults(minimal_scene):
    minimal_scene.pop("lights")
    assert validate_scene(minimal_scene)["scene"]["lights"] == DEFAULT_LIGHTS


def test_all_invalid_lights_fall_back_to_defaults(minimal_scene):
    minimal_scene["lights"] = [{"type": "laser", "intensity": 1.0}]
    assert validate_scene(minimal_scene)["scene"]["lights"] == DEFAULT_LIGHTS


def test_missing_camera_falls_back_to_default(minimal_scene):
    minimal_scene.pop("camera")
    assert validate_scene(minimal_scene)["scene"]["camera"] == DEFAULT_CAMERA


def test_out_of_range_fov_falls_back(minimal_scene):
    minimal_scene["camera"]["fov"] = 500
    assert validate_scene(minimal_scene)["scene"]["camera"]["fov"] == DEFAULT_CAMERA["fov"]


# ── constant sets (guard against silent widening) ────────────────────────────

def test_valid_type_sets_are_exact():
    assert VALID_OBJECT_TYPES == {"primitive", "mesh"}
    assert VALID_GEOM_TYPES == {
        "sphere", "box", "cylinder", "plane", "ring", "capsule", "torus"}
    assert VALID_ANIM_TYPES == {"none", "orbit", "spin", "physics"}
    assert VALID_PHYSICS_TYPES == {"gravity", "shm", "pendulum", "projectile"}
    assert VALID_LIGHT_TYPES == {"ambient", "directional", "point", "spot"}
