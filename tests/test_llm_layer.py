import json
from unittest.mock import MagicMock, patch

from llm.scene_schema import SceneSchema
from llm.prompt_templates import build_system_prompt, build_refinement_prompt
from llm.context_manager import ContextManager


VALID_OBJECT = {
    "id": "sun",
    "type": "sphere",
    "position": [0.0, 0.0, 0.0],
    "color": [1.0, 0.84, 0.0],
    "animation": "none",
    "orbit_center": [0.0, 0.0, 0.0],
    "orbit_speed": 0.0,
}

VALID_SCENE = {"objects": [VALID_OBJECT]}


# ─── scene_schema tests ────────────────────────────────────────────────────────

def test_valid_scene_passes():
    scene = SceneSchema.model_validate(VALID_SCENE)
    assert len(scene.objects) == 1
    assert scene.objects[0].type == "sphere"
    print("OK: valid scene passes validation")


def test_invalid_type_rejected():
    bad = {**VALID_OBJECT, "type": "triangle"}
    try:
        SceneSchema.model_validate({"objects": [bad]})
        raise AssertionError("Should have raised ValidationError")
    except Exception as e:
        assert "type" in str(e).lower() or "literal" in str(e).lower()
    print("OK: invalid object type rejected")


def test_invalid_animation_rejected():
    bad = {**VALID_OBJECT, "animation": "spin"}
    try:
        SceneSchema.model_validate({"objects": [bad]})
        raise AssertionError("Should have raised ValidationError")
    except Exception:
        pass
    print("OK: invalid animation rejected")


def test_empty_objects_rejected():
    try:
        SceneSchema.model_validate({"objects": []})
        raise AssertionError("Should have raised ValidationError")
    except Exception:
        pass
    print("OK: empty objects list rejected")


def test_position_wrong_length_rejected():
    bad = {**VALID_OBJECT, "position": [0.0, 0.0]}
    try:
        SceneSchema.model_validate({"objects": [bad]})
        raise AssertionError("Should have raised ValidationError")
    except Exception:
        pass
    print("OK: position with wrong length rejected")


def test_max_objects_enforced():
    objects = [{**VALID_OBJECT, "id": f"obj{i}"} for i in range(21)]
    try:
        SceneSchema.model_validate({"objects": objects})
        raise AssertionError("Should have raised ValidationError")
    except Exception:
        pass
    print("OK: more than 20 objects rejected")


def test_max_objects_boundary_passes():
    objects = [{**VALID_OBJECT, "id": f"obj{i}"} for i in range(20)]
    scene = SceneSchema.model_validate({"objects": objects})
    assert len(scene.objects) == 20
    print("OK: exactly 20 objects passes")


def test_all_types_accepted():
    for obj_type in ("sphere", "cube", "cylinder", "ring", "label"):
        obj = {**VALID_OBJECT, "id": obj_type, "type": obj_type}
        scene = SceneSchema.model_validate({"objects": [obj]})
        assert scene.objects[0].type == obj_type
    print("OK: all valid types accepted")


def test_both_animations_accepted():
    for anim in ("none", "orbit"):
        obj = {**VALID_OBJECT, "animation": anim}
        scene = SceneSchema.model_validate({"objects": [obj]})
        assert scene.objects[0].animation == anim
    print("OK: both animation values accepted")


# ─── prompt_templates tests ────────────────────────────────────────────────────

def test_system_prompt_contains_schema_fields():
    prompt = build_system_prompt()
    for field in ("sphere", "cube", "cylinder", "ring", "label", "orbit_center", "orbit_speed", "animation"):
        assert field in prompt, f"Missing field in system prompt: {field}"
    print("OK: system prompt contains all SCENE_GRAMMAR fields")


def test_system_prompt_no_markdown():
    prompt = build_system_prompt()
    assert "```" not in prompt
    print("OK: system prompt contains no markdown code fences")


def test_refinement_prompt_contains_command_and_scene():
    prompt = build_refinement_prompt(VALID_SCENE, "make it bigger")
    assert "make it bigger" in prompt
    assert "sun" in prompt
    print("OK: refinement prompt contains command and previous scene")


# ─── context_manager tests ─────────────────────────────────────────────────────

def test_context_manager_add_and_last_scene():
    cm = ContextManager()
    cm.add("show me the solar system", VALID_SCENE)
    assert cm.last_scene() == VALID_SCENE
    print("OK: last_scene() returns most recent scene")


def test_context_manager_empty_last_scene():
    cm = ContextManager()
    assert cm.last_scene() is None
    print("OK: last_scene() returns None when empty")


def test_context_manager_maxlen_enforced():
    cm = ContextManager()
    for i in range(7):
        cm.add(f"command {i}", {"objects": [{**VALID_OBJECT, "id": f"obj{i}"}]})
    assert len(cm._history) == 5
    print("OK: deque maxlen=5 enforced")


def test_context_manager_compression():
    cm = ContextManager()
    for i in range(5):
        cm.add(f"command {i+1}", {"objects": [{**VALID_OBJECT, "id": f"obj{i}"}]})
    prompt = cm.build_refinement_prompt("rotate it")
    assert "compressed" in prompt
    assert "rotate it" in prompt
    compressed_count = prompt.count("compressed")
    assert compressed_count == 2, f"Expected 2 compressed entries, got {compressed_count}"
    print(f"OK: {compressed_count} entries compressed, recent 3 kept full")


def test_context_manager_no_compression_below_threshold():
    cm = ContextManager()
    for i in range(3):
        cm.add(f"command {i+1}", {"objects": [{**VALID_OBJECT, "id": f"obj{i}"}]})
    prompt = cm.build_refinement_prompt("add a moon")
    assert "compressed" not in prompt
    print("OK: no compression when history has 3 or fewer entries")


# ─── gemini_client fallback test (no real API call) ───────────────────────────

def test_generate_scene_returns_fallback_on_api_error():
    from llm.gemini_client import FALLBACK_SCENE

    with patch("llm.gemini_client._call_gemini", side_effect=Exception("quota exceeded")):
        from llm.gemini_client import generate_scene
        result = generate_scene("show me anything", None)
        assert result == FALLBACK_SCENE
    print("OK: generate_scene returns FALLBACK_SCENE on API error")


def test_generate_scene_returns_valid_scene_on_success():
    valid_response = json.dumps(VALID_SCENE)

    with patch("llm.gemini_client._call_gemini", return_value=valid_response):
        from llm.gemini_client import generate_scene
        result = generate_scene("show me the solar system", None)
        assert "objects" in result
        assert len(result["objects"]) == 1
    print("OK: generate_scene returns validated scene on success")


def test_generate_scene_retries_on_validation_error():
    invalid_response = json.dumps({"objects": []})
    valid_response = json.dumps(VALID_SCENE)
    call_count = {"n": 0}

    def fake_call(prompt):
        call_count["n"] += 1
        return invalid_response if call_count["n"] == 1 else valid_response

    with patch("llm.gemini_client._call_gemini", side_effect=fake_call):
        from llm.gemini_client import generate_scene
        result = generate_scene("show me something", None)
        assert call_count["n"] == 2
        assert result == VALID_SCENE
    print("OK: generate_scene retried once and succeeded on second attempt")


# ─── runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    suites = {
        "scene_schema": [
            test_valid_scene_passes,
            test_invalid_type_rejected,
            test_invalid_animation_rejected,
            test_empty_objects_rejected,
            test_position_wrong_length_rejected,
            test_max_objects_enforced,
            test_max_objects_boundary_passes,
            test_all_types_accepted,
            test_both_animations_accepted,
        ],
        "prompt_templates": [
            test_system_prompt_contains_schema_fields,
            test_system_prompt_no_markdown,
            test_refinement_prompt_contains_command_and_scene,
        ],
        "context_manager": [
            test_context_manager_add_and_last_scene,
            test_context_manager_empty_last_scene,
            test_context_manager_maxlen_enforced,
            test_context_manager_compression,
            test_context_manager_no_compression_below_threshold,
        ],
        "gemini_client (mocked)": [
            test_generate_scene_returns_fallback_on_api_error,
            test_generate_scene_returns_valid_scene_on_success,
            test_generate_scene_retries_on_validation_error,
        ],
    }

    total = 0
    for suite_name, tests in suites.items():
        print(f"\n=== {suite_name} ===")
        for test_fn in tests:
            test_fn()
            total += 1

    print(f"\nAll {total} tests passed.")
