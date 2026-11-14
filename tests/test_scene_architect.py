"""
pipeline/scene_architect.py — the deterministic post-processors.

The three LLM passes themselves are not tested here (they need Vertex AI); what
IS tested is everything that runs on their output, because those functions are
the last line of defence against malformed model output reaching the validator.
`_default_lighting_camera` matters most: it is the no-LLM fallback that keeps a
good scene alive when pass 7c fails.
"""
from __future__ import annotations

import pytest

from pipeline.scene_architect import (
    _clean_json,
    _default_lighting_camera,
    _fix_plane_intersections,
    _is_numeric_xyz,
    _normalize_object,
    _parse_json,
    _sanitize_mesh_paths,
)


# ── JSON extraction from LLM text ────────────────────────────────────────────

def test_parse_json_plain():
    assert _parse_json('{"name": "x"}') == {"name": "x"}


def test_parse_json_strips_markdown_fences():
    raw = 'Here you go:\n```json\n{"name": "Scene"}\n```\nHope that helps!'
    assert _parse_json(raw) == {"name": "Scene"}


def test_parse_json_tolerates_trailing_commas():
    assert _parse_json('{"a": 1, "b": [1, 2,],}') == {"a": 1, "b": [1, 2]}


def test_parse_json_strips_line_comments():
    assert _parse_json('{\n  // the name\n  "name": "x"\n}') == {"name": "x"}


def test_parse_json_returns_none_on_garbage():
    assert _parse_json("no json here at all") is None
    assert _parse_json("") is None


def test_clean_json_is_idempotent():
    text = '{"a": 1}'
    assert _clean_json(_clean_json(text)) == _clean_json(text)


# ── vector helper ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "value,expected",
    [
        ([1, 2, 3], True), ([1.5, -2.0, 0], True),
        ([1, 2], False), ([1, 2, 3, 4], False),
        (["a", 2, 3], False), (None, False), ("1,2,3", False),
        ([float("nan"), 1, 2], False),
    ],
)
def test_is_numeric_xyz(value, expected):
    assert _is_numeric_xyz(value) is expected


# ── object normalization ─────────────────────────────────────────────────────

def test_scalar_scale_expanded_to_vector():
    assert _normalize_object({"scale": 3})["scale"] == [3, 3, 3]


def test_invalid_position_reset_to_origin():
    assert _normalize_object({"position": "middle"})["position"] == [0.0, 0.0, 0.0]


def test_missing_animation_filled_in():
    assert _normalize_object({})["animation"] == {"type": "none"}


def test_named_color_replaced_for_primitive():
    obj = _normalize_object({"type": "primitive", "material": {"color": "red"}})
    assert obj["material"]["color"] == "#888888"


def test_named_color_replaced_with_white_for_mesh():
    """Meshes carry embedded textures — any tint destroys them."""
    obj = _normalize_object({"type": "mesh", "material": {"color": "red"}})
    assert obj["material"]["color"] == "#ffffff"


def test_cylinder_with_non_numeric_endpoints_is_dropped():
    obj = _normalize_object({
        "id": "bad_tube", "type": "primitive",
        "geometry": {"type": "cylinder", "from": ["sin(x)", 0, 0], "to": [1, 1, 1]},
    })
    assert obj.get("_drop") is True


# ── mesh path sanitization ───────────────────────────────────────────────────

def test_unverified_mesh_becomes_primitive_fallback():
    scene = {"objects": [{
        "id": "ghost", "type": "mesh", "model": "/assets/meshes/made_up.glb",
        "position": [0, 0, 0],
    }]}
    out = _sanitize_mesh_paths(scene, allowed=set())
    obj = out["objects"][0]
    assert obj["type"] == "primitive"
    assert obj["geometry"]["type"] == "sphere"


def test_verified_mesh_kept_as_mesh():
    path = "/assets/meshes/abstract/brain.glb"
    scene = {"objects": [{
        "id": "brain", "type": "mesh", "model": path, "position": [0, 0, 0],
    }]}
    out = _sanitize_mesh_paths(scene, allowed={path})
    assert out["objects"][0]["type"] == "mesh"
    assert out["objects"][0]["model"] == path


def test_label_and_description_survive_mesh_fallback():
    """
    Educational text must not be lost when a hallucinated mesh is downgraded —
    the fallback rebuilds the object dict from scratch, so each preserved field
    needs explicit carry-through.
    """
    scene = {"objects": [{
        "id": "x", "type": "mesh", "model": "/nope.glb", "position": [0, 0, 0],
        "label": "Heart", "description": "Pumps blood.", "parent": "torso",
    }]}
    obj = _sanitize_mesh_paths(scene, allowed=set())["objects"][0]
    assert obj["label"] == "Heart"
    assert obj["description"] == "Pumps blood."
    assert obj["parent"] == "torso"


# ── plane intersection repair ────────────────────────────────────────────────

def test_plane_pushed_below_lowest_object():
    scene = {"objects": [
        {"id": "floor", "type": "primitive",
         "geometry": {"type": "plane", "width": 10, "height": 10},
         "position": [0, 0, 0]},
        {"id": "ball", "type": "primitive",
         "geometry": {"type": "sphere", "radius": 1}, "position": [0, 0.5, 0]},
    ]}
    out = _fix_plane_intersections(scene)
    plane = next(o for o in out["objects"] if o["id"] == "floor")
    assert plane["position"][1] < 0.5


def test_scene_without_planes_untouched():
    scene = {"objects": [
        {"id": "ball", "type": "primitive",
         "geometry": {"type": "sphere", "radius": 1}, "position": [0, 3, 0]},
    ]}
    assert _fix_plane_intersections(scene)["objects"][0]["position"] == [0, 3, 0]


# ── deterministic lighting/camera fallback (pass 7c failure path) ────────────

def test_default_lighting_camera_shape():
    objects = [
        {"id": "a", "position": [0, 0, 0]},
        {"id": "b", "position": [10, 0, 0]},
    ]
    out = _default_lighting_camera(objects)
    assert {"name", "summary", "lights", "camera"} <= set(out)
    kinds = [light["type"] for light in out["lights"]]
    assert kinds == ["ambient", "directional", "point"]
    assert out["camera"]["fov"] == 60


def test_default_camera_frames_the_bounding_box():
    """Camera must sit outside the scene, looking at its centre."""
    objects = [{"id": "a", "position": [0, 0, 0]}, {"id": "b", "position": [20, 0, 0]}]
    cam = _default_lighting_camera(objects)["camera"]
    centre_x = 10.0
    assert cam["target"] == [centre_x, 0.0, 0.0]
    assert cam["position"][2] > 0          # pulled back along +Z
    assert cam["position"][1] > 0          # raised above the scene


def test_default_lighting_handles_empty_and_malformed_objects():
    """Must never divide by zero or raise — it is the last-resort path."""
    for objects in ([], [{"id": "x"}], [{"id": "y", "position": "bad"}]):
        out = _default_lighting_camera(objects)
        assert out["camera"]["target"] == [0.0, 0.0, 0.0]
        assert len(out["lights"]) == 3


def test_default_lighting_enforces_minimum_radius():
    """Co-located objects must not produce a camera inside the geometry."""
    objects = [{"id": "a", "position": [0, 0, 0]}, {"id": "b", "position": [0, 0, 0]}]
    cam = _default_lighting_camera(objects)["camera"]
    assert cam["position"][2] >= 6.0       # radius floored at 3.0, times 2.0


def test_directional_light_casts_shadow():
    out = _default_lighting_camera([{"id": "a", "position": [0, 0, 0]}])
    directional = next(l for l in out["lights"] if l["type"] == "directional")
    assert directional["castShadow"] is True


# ── orbital-plane rule (regression: 2026-08-06) ──────────────────────────────

def _architect_source() -> str:
    from pathlib import Path
    return (Path(__file__).resolve().parents[1] / "pipeline" /
            "scene_architect.py").read_text(encoding="utf-8")


def test_prompt_states_the_orbital_plane_rule():
    """
    torusGeometry is built in the XY plane but orbit animation moves objects in
    XZ (ThreeScene uses only dx/dz). Without rotation [90,0,0] a shell ring
    stands vertical while its electrons sweep flat through it — which is exactly
    what a carbon-atom scene produced.
    """
    src = _architect_source()
    assert "ORBITAL PLANE" in src
    assert "[90, 0, 0]" in src or "[90,0,0]" in src


def test_prompt_ties_shell_radius_to_electron_orbit_radius():
    src = _architect_source()
    assert "radius EXACTLY equal to the" in src


def test_prompt_specifies_phase_spreading():
    """Electrons sharing a shell must get distinct phases or they stack up."""
    src = _architect_source()
    assert "phase" in src and "4.71239" in src
