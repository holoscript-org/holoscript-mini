"""
Cross-language schema drift guard.

pipeline/scene_validator.py is a hand-maintained port of gui/lib/sceneFactory.ts.
Section 22.1 of docs/CODEBASE_DOCUMENTATION.md calls this the project's critical
sync risk, and it has already bitten once: adding `summary`/`description`
required six separate edits across the two files, and missing any one of them
drops the field silently at that boundary with no error anywhere.

These tests parse the TypeScript source textually (no Node required) and assert
the two sides still agree. They are intentionally strict — a deliberate schema
change should fail here and be updated in both places.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from pipeline.scene_validator import (
    VALID_ANIM_TYPES,
    VALID_GEOM_TYPES,
    VALID_LIGHT_TYPES,
    VALID_OBJECT_TYPES,
    VALID_PHYSICS_TYPES,
)

_TS = Path(__file__).resolve().parents[1] / "gui" / "lib" / "sceneFactory.ts"


@pytest.fixture(scope="module")
def ts_source() -> str:
    assert _TS.exists(), f"sceneFactory.ts not found at {_TS}"
    return _TS.read_text(encoding="utf-8")


def _string_union(source: str, type_name: str) -> set[str]:
    """Extract `export type X = "a" | "b" | "c"` into {"a","b","c"}."""
    match = re.search(
        rf'export type {type_name}\s*=\s*([^\n]+(?:\n\s*\|[^\n]+)*)', source
    )
    assert match, f"could not find `export type {type_name}` in sceneFactory.ts"
    return set(re.findall(r'"([^"]+)"', match.group(1)))


def _const_array(source: str, const_name: str) -> set[str]:
    """Extract `const X = ["a","b"]` (optionally with a type annotation)."""
    match = re.search(rf'const {const_name}[^=]*=\s*\[(.*?)\]', source, re.S)
    assert match, f"could not find `const {const_name}` in sceneFactory.ts"
    return set(re.findall(r'"([^"]+)"', match.group(1)))


# Compare against the exported VALID_* arrays rather than the type unions:
# those arrays are what validateScene() actually branches on at runtime, so
# they are the real contract. (PrimitiveGeomType is also declared as a union,
# but only the array is consulted during validation.)

@pytest.mark.parametrize(
    "ts_const,py_set",
    [
        ("VALID_OBJECT_TYPES",  VALID_OBJECT_TYPES),
        ("VALID_GEOM_TYPES",    VALID_GEOM_TYPES),
        ("VALID_ANIM_TYPES",    VALID_ANIM_TYPES),
        ("VALID_PHYSICS_TYPES", VALID_PHYSICS_TYPES),
        ("VALID_LIGHT_TYPES",   VALID_LIGHT_TYPES),
    ],
)
def test_valid_type_arrays_match_python(ts_source, ts_const, py_set):
    assert _const_array(ts_source, ts_const) == py_set, (
        f"{ts_const} has drifted between sceneFactory.ts and scene_validator.py"
    )


def test_type_unions_agree_with_their_arrays(ts_source):
    """The TS union and the TS array must also agree with each other."""
    assert _string_union(ts_source, "ObjectType") == _const_array(ts_source, "VALID_OBJECT_TYPES")
    assert _string_union(ts_source, "LightType") == _const_array(ts_source, "VALID_LIGHT_TYPES")
    assert _string_union(ts_source, "AnimationType") == _const_array(ts_source, "VALID_ANIM_TYPES")
    assert _string_union(ts_source, "PhysicsType") == _const_array(ts_source, "VALID_PHYSICS_TYPES")
    assert _string_union(ts_source, "PrimitiveGeomType") == _const_array(ts_source, "VALID_GEOM_TYPES")


# ── field-level parity for the fields that have actually drifted ─────────────

@pytest.mark.parametrize("field", ["label", "description"])
def test_object_optional_text_fields_present_on_both_sides(ts_source, field):
    """
    Each of these must appear in the TS SceneObject interface AND be passed
    through by the Python validator's object rebuild.
    """
    assert re.search(rf"\n\s*{field}\?:\s*string", ts_source), \
        f"SceneObject.{field} missing from sceneFactory.ts"

    py = (Path(__file__).resolve().parents[1] / "pipeline" / "scene_validator.py"
          ).read_text(encoding="utf-8")
    assert f'"{field}"' in py, f"{field} not handled in scene_validator.py"


def test_scene_summary_present_on_both_sides(ts_source):
    assert re.search(r"\n\s*summary\?:\s*string", ts_source), \
        "SceneDef.summary missing from sceneFactory.ts"
    py = (Path(__file__).resolve().parents[1] / "pipeline" / "scene_validator.py"
          ).read_text(encoding="utf-8")
    assert '"summary"' in py, "summary not passed through in scene_validator.py"


def test_mirror_status_comment_exists():
    """The port carries a sync date; keep it as the human-facing reminder."""
    py = (Path(__file__).resolve().parents[1] / "pipeline" / "scene_validator.py"
          ).read_text(encoding="utf-8")
    assert "MIRROR STATUS" in py
