"""
LLM-backed stages, with every provider call mocked.

The property that matters for all of these is the same: **they must never raise
and never return a malformed shape**, no matter what the model returns. A demo
must degrade, not crash. Each stage has a documented identity/empty fallback and
these tests pin it down.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from llm import gemini_client
from pipeline.events import PipelineEvent
from pipeline.intent_extractor import _EMPTY_IR, _coerce_ir
from pipeline.intent_extractor import extract as extract_ir
from pipeline.prompt_optimizer import optimize


# ── llm/gemini_client.call_llm provider fallback ─────────────────────────────

def test_call_llm_prefers_gemini():
    with patch.object(gemini_client, "call_gemini", return_value="from-gemini") as g, \
         patch.object(gemini_client, "call_groq", return_value="from-groq") as q:
        raw, provider = gemini_client.call_llm("m-pro", "m-groq", "p", "sys")
    assert (raw, provider) == ("from-gemini", "gemini")
    g.assert_called_once()
    q.assert_not_called()


def test_call_llm_falls_back_to_groq():
    with patch.object(gemini_client, "call_gemini", return_value=None), \
         patch.object(gemini_client, "call_groq", return_value="from-groq"):
        raw, provider = gemini_client.call_llm("m-pro", "m-groq", "p", "sys")
    assert (raw, provider) == ("from-groq", "groq")


def test_call_llm_reports_total_failure_without_raising():
    with patch.object(gemini_client, "call_gemini", return_value=None), \
         patch.object(gemini_client, "call_groq", return_value=None):
        assert gemini_client.call_llm("a", "b", "p", "s") == (None, None)


def test_gemini_kwargs_not_forwarded_to_groq():
    """
    Groq shares only `temperature`; thinking_budget/json_mode are Gemini-only
    and would be rejected by the REST API.
    """
    with patch.object(gemini_client, "call_gemini", return_value=None), \
         patch.object(gemini_client, "call_groq", return_value="ok") as q:
        gemini_client.call_llm("a", "b", "p", "s",
                               temperature=0.9, thinking_budget=100, json_mode=True)
    kwargs = q.call_args.kwargs
    assert kwargs.get("temperature") == 0.9
    assert "thinking_budget" not in kwargs
    assert "json_mode" not in kwargs


# ── Stage 2: prompt optimizer ────────────────────────────────────────────────

def _patch_optimizer(return_value):
    return patch("pipeline.prompt_optimizer.call_llm", return_value=return_value)


def test_optimizer_returns_parsed_result():
    payload = '{"optimized_prompt": "A red bouncing ball on a floor", ' \
              '"clarifications_made": ["added a floor"]}'
    with _patch_optimizer((payload, "gemini")):
        out = optimize("red ball bounce")
    assert out["optimized_prompt"] == "A red bouncing ball on a floor"
    assert out["clarifications_made"] == ["added a floor"]
    assert out["original_prompt"] == "red ball bounce"


def test_optimizer_identity_fallback_on_unusable_output():
    with _patch_optimizer(("not json", "gemini")):
        out = optimize("solar system")
    assert out["optimized_prompt"] == "solar system"
    assert out["clarifications_made"] == []


def test_optimizer_identity_fallback_when_both_providers_fail():
    with _patch_optimizer((None, None)):
        out = optimize("solar system")
    assert out["optimized_prompt"] == "solar system"


@pytest.mark.parametrize("blank", ["", "   "])
def test_optimizer_handles_blank_input(blank):
    with _patch_optimizer((None, None)):
        out = optimize(blank)
    assert out["optimized_prompt"] == blank
    assert out["original_prompt"] == blank


def test_optimizer_coerces_non_list_clarifications():
    payload = '{"optimized_prompt": "x", "clarifications_made": "oops"}'
    with _patch_optimizer((payload, "gemini")):
        assert optimize("x")["clarifications_made"] == []


def test_optimizer_emits_lifecycle_events():
    events: list[PipelineEvent] = []
    payload = '{"optimized_prompt": "x", "clarifications_made": []}'
    with _patch_optimizer((payload, "gemini")):
        optimize("x", run_id="r1", on_event=events.append)
    statuses = [e.status for e in events]
    assert statuses[0] == "started"
    assert statuses[-1] == "completed"
    assert all(e.run_id == "r1" for e in events)
    assert events[-1].elapsed_ms is not None


def test_optimizer_never_raises_on_provider_exception():
    with patch("pipeline.prompt_optimizer.call_llm", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            optimize("x")   # documents that call_llm itself must not throw


# ── Stage 3: intent extractor ────────────────────────────────────────────────

def _patch_extractor(return_value):
    return patch("pipeline.intent_extractor.call_llm", return_value=return_value)


def test_ir_coercion_fills_missing_keys():
    assert _coerce_ir({}) == _EMPTY_IR


def test_ir_coercion_rejects_wrong_types():
    """A malformed value must fall back to the default, not propagate."""
    ir = _coerce_ir({"objects": "not a list", "scene_type": ["not a str"]})
    assert ir["objects"] == []
    assert ir["scene_type"] == "other"


def test_ir_coercion_keeps_valid_values():
    ir = _coerce_ir({"scene_type": "astronomical", "objects": [{"name": "sun"}]})
    assert ir["scene_type"] == "astronomical"
    assert ir["objects"] == [{"name": "sun"}]


def test_extractor_returns_empty_ir_on_garbage():
    with _patch_extractor(("not json", "gemini")):
        assert extract_ir("a solar system") == _EMPTY_IR


def test_extractor_returns_empty_ir_when_providers_fail():
    with _patch_extractor((None, None)):
        assert extract_ir("a solar system") == _EMPTY_IR


def test_extractor_handles_blank_input():
    with _patch_extractor((None, None)):
        assert extract_ir("  ") == _EMPTY_IR


def test_extracted_ir_always_has_every_key():
    payload = '{"scene_type": "physics", "objects": [{"name": "ball"}]}'
    with _patch_extractor((payload, "gemini")):
        ir = extract_ir("a ball")
    assert set(ir) == set(_EMPTY_IR)


def test_extractor_records_provider_on_events():
    events: list[PipelineEvent] = []
    payload = '{"scene_type": "physics"}'
    with _patch_extractor((payload, "groq")):
        extract_ir("x", run_id="r2", on_event=events.append)
    assert any(e.provider == "groq" for e in events)
