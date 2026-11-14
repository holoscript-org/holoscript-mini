"""
pipeline/pipeline_runner.py pure helpers + pipeline/events.py contract.

The candidate-building path is what feeds Poly Pizza, and every query it emits
can become a permanent MongoDB concept id. The `_extract_candidate_phrases`
fallback-loop test below guards a leak found on 2026-08-05: a bare
`token not in SEARCH_STOP` check passes multi-word tokens like
"basketball falling" straight through, because the *phrase* is obviously not a
member of the stopword set.
"""
from __future__ import annotations

import pytest

from pipeline.events import (
    COMPLETED,
    FAILED,
    OUTPUT,
    STARTED,
    PipelineEvent,
    cli_event_printer,
    make_emitter,
    new_run_id,
)
from pipeline.pipeline_runner import (
    _build_live_candidates,
    _extract_candidate_phrases,
    _is_refinement,
    _strip_none,
)


# ── candidate phrase extraction ──────────────────────────────────────────────

def test_glued_phrase_never_emitted():
    """The original bug: 'saturn and its rings' -> 'saturn its rings'."""
    out = _extract_candidate_phrases(
        "solar system with saturn and its rings", ["saturn", "its", "rings"]
    )
    assert "saturn its rings" not in out
    assert "its rings" not in out


def test_multiword_unresolved_token_with_motion_verb_rejected():
    """
    Regression guard. The parser can hand back multi-word tokens it built
    itself; "basketball falling" must not survive the single-token fallback.
    """
    out = _extract_candidate_phrases(
        "A basketball falling and bouncing on a hardwood floor",
        ["basketball falling", "bouncing", "hardwood"],
    )
    assert "basketball falling" not in out
    assert "hardwood" in out


def test_prose_fragment_token_rejected():
    out = _extract_candidate_phrases(
        "A google-style technical office scene showing a desk",
        ["google-style technical office", "desk"],
    )
    assert "google-style technical office" not in out
    assert "desk" in out


def test_real_compound_preserved():
    out = _extract_candidate_phrases(
        "show me a wrist watch on a table", ["wrist", "watch", "table"]
    )
    assert "wrist watch" in out


def test_only_unresolved_words_form_phrases():
    """A phrase whose words are already known to the KB must not be searched."""
    out = _extract_candidate_phrases("wrist watch on a table", ["table"])
    assert "wrist watch" not in out
    assert out == ["table"]


# ── live-candidate gating ────────────────────────────────────────────────────

def test_primitive_only_concepts_never_searched():
    """
    Astronomical concepts must be built from primitives. Downloading a mesh for
    them creates a rule conflict the architect resolves the wrong way.
    """
    cands = _build_live_candidates(
        {"objects": ["saturn", "rings", "earth"]}, [], [],
        "solar system with saturn and its rings",
    )
    assert cands == []


def test_real_objects_still_searched():
    cands = _build_live_candidates(
        {"objects": ["wrist watch"]}, [], [], "show me a wrist watch"
    )
    assert [c["concept"] for c in cands] == ["wrist watch"]


def test_already_verified_assets_not_researched():
    cands = _build_live_candidates(
        {"objects": ["dolphin"]},
        [],
        [{"concept": "dolphin", "path": "/assets/meshes/abstract/dolphin.glb"}],
        "a dolphin",
    )
    assert cands == []


def test_candidates_capped_at_four():
    intent = {"objects": [f"widget{i}" for i in range(12)]}
    cands = _build_live_candidates(intent, [], [], "many widgets")
    assert len(cands) <= 4


def test_candidates_deduplicated():
    cands = _build_live_candidates(
        {"objects": ["lamp", "lamp"]}, ["lamp"], [], "a lamp and a lamp"
    )
    assert len(cands) == len({c["concept"] for c in cands})


# ── misc helpers ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "text,expected",
    [
        ("add a moon", True), ("remove the sun", True), ("make it bigger", True),
        ("now rotate it", True), ("solar system", False), ("a bouncing ball", False),
    ],
)
def test_is_refinement(text, expected):
    assert _is_refinement(text) is expected


def test_strip_none_removes_null_keys():
    out = _strip_none({"a": 1, "b": None, "c": {"d": None, "e": 2}, "f": [None, 3]})
    assert "b" not in out
    assert "d" not in out["c"]
    assert out["c"]["e"] == 2
    assert out["a"] == 1


def test_strip_none_preserves_falsy_non_none():
    out = _strip_none({"zero": 0, "empty": "", "false": False, "none": None})
    assert out["zero"] == 0 and out["empty"] == "" and out["false"] is False
    assert "none" not in out


# ── event contract ───────────────────────────────────────────────────────────

def test_run_ids_are_unique_hex():
    ids = {new_run_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(len(i) == 32 and int(i, 16) >= 0 for i in ids)


def test_emitter_is_a_noop_without_listener():
    """
    Call sites emit unconditionally. With on_event=None this must cost nothing
    and never raise — that is why adding events required no branching in
    existing stage code.
    """
    emit = make_emitter("run1", None)
    emit("stage", STARTED, "Stage")  # must not raise


def test_emitter_forwards_all_fields():
    captured: list[PipelineEvent] = []
    emit = make_emitter("run-abc", captured.append)
    emit("architect_layout", OUTPUT, "Layout",
         payload={"objects": 3}, provider="gemini", elapsed_ms=1234)

    (event,) = captured
    assert event.run_id == "run-abc"
    assert event.stage == "architect_layout"
    assert event.status == OUTPUT
    assert event.label == "Layout"
    assert event.payload == {"objects": 3}
    assert event.provider == "gemini"
    assert event.elapsed_ms == 1234
    assert event.timestamp > 0


def test_event_optional_fields_default_to_none():
    captured: list[PipelineEvent] = []
    make_emitter("r", captured.append)("s", COMPLETED, "S")
    assert captured[0].payload is None
    assert captured[0].provider is None
    assert captured[0].elapsed_ms is None


def test_status_constants_are_the_wire_values():
    assert (STARTED, OUTPUT, COMPLETED, FAILED) == (
        "started", "output", "completed", "failed")


def test_cli_event_printer_outputs_label_and_status(capsys):
    cli_event_printer(PipelineEvent(
        run_id="r", stage="s", status="completed", label="Validate",
        elapsed_ms=42, provider="groq",
    ))
    out = capsys.readouterr().out
    assert "Validate" in out and "completed" in out and "42" in out and "groq" in out
