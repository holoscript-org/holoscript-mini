"""
pipeline/live_search.py — the relevance gate, with all network/DB calls mocked.

This is the write path into the knowledge base: `_remember_asset_query()` uses
the raw search query as the MongoDB `_id`, so anything that gets past the gate
becomes a permanent concept. Before the gate existed, `results[:1]` was accepted
unconditionally — Poly Pizza always ranks *something* first and never reports
"no match", which is how 'saturn' ended up mapped to a generic Planets model.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import pipeline.live_search as ls


@pytest.fixture
def poly_env(monkeypatch, tmp_path):
    """Neutralize every external dependency of fetch_live_assets()."""
    monkeypatch.setattr(ls, "_MESHES", tmp_path)
    monkeypatch.setattr(ls, "_polypizza_headers", lambda: {"x-auth-token": "test"})
    monkeypatch.setattr(ls, "lookup_concept", lambda _n: None)
    monkeypatch.setattr(ls, "get_cached_asset", lambda _n: None)
    monkeypatch.setattr(ls, "register_concept", lambda **kw: None)
    monkeypatch.setattr(ls, "embed_concept", lambda *a, **k: None)
    return tmp_path


def _result(title: str, tags: list[str] | None = None) -> dict:
    return {
        "Title": title,
        "Download": "https://example.test/model.glb",
        "Tags": tags or [],
        "Creator": {"Username": "tester"},
        "License": "CC-BY",
    }


def test_irrelevant_top_result_rejected(poly_env):
    """'saturn' -> a generic 'Planets' model must not be registered."""
    with patch.object(ls, "_search_polypizza", return_value=[_result("Planets", ["space"])]), \
         patch.object(ls, "_download_file", return_value=True) as dl:
        found = ls.fetch_live_assets([{"concept": "saturn", "category": "planets"}])

    assert found == []
    dl.assert_not_called()          # nothing downloaded


def test_relevant_result_accepted(poly_env):
    with patch.object(ls, "_search_polypizza", return_value=[_result("Dolphin", ["sea"])]), \
         patch.object(ls, "_download_file", return_value=True):
        found = ls.fetch_live_assets([{"concept": "dolphin", "category": "abstract"}])

    assert len(found) == 1
    assert found[0]["concept"] == "dolphin"
    assert found[0]["source"] == "download"


def test_gate_skips_past_bad_results_to_a_good_one(poly_env):
    """
    The loop must not stop at the first result. Before the fix it sliced
    results[:1]; now it scans until `max_per_concept` relevant hits are found.
    """
    results = [
        _result("Onion Rings", ["food"]),
        _result("Traffic Cone", ["road"]),
        _result("Wristwatch", ["time"]),
    ]
    with patch.object(ls, "_search_polypizza", return_value=results), \
         patch.object(ls, "_download_file", return_value=True):
        found = ls.fetch_live_assets([{"concept": "wrist watch", "category": "abstract"}])

    assert len(found) == 1
    assert found[0]["concept"] == "wrist watch"


def test_max_per_concept_respected(poly_env):
    results = [_result("Dolphin One", ["dolphin"]), _result("Dolphin Two", ["dolphin"])]
    with patch.object(ls, "_search_polypizza", return_value=results), \
         patch.object(ls, "_download_file", return_value=True):
        found = ls.fetch_live_assets(
            [{"concept": "dolphin", "category": "abstract"}], max_per_concept=1
        )
    assert len(found) == 1


def test_compound_word_title_accepted(poly_env):
    """'basketball' vs a model titled 'Basket Ball' — same thing, no shared token."""
    with patch.object(ls, "_search_polypizza", return_value=[_result("Basket Ball", [])]), \
         patch.object(ls, "_download_file", return_value=True):
        found = ls.fetch_live_assets([{"concept": "basketball", "category": "abstract"}])
    assert len(found) == 1


def test_all_results_irrelevant_registers_nothing(poly_env):
    results = [_result("Harp", ["music"]), _result("Piano", ["music"])]
    with patch.object(ls, "_search_polypizza", return_value=results), \
         patch.object(ls, "_download_file", return_value=True), \
         patch.object(ls, "register_concept") as reg:
        found = ls.fetch_live_assets([{"concept": "harmonic motion", "category": "abstract"}])

    assert found == []
    reg.assert_not_called()         # nothing written to MongoDB


def test_search_exception_is_swallowed(poly_env):
    with patch.object(ls, "_search_polypizza", side_effect=RuntimeError("api down")):
        assert ls.fetch_live_assets([{"concept": "dolphin", "category": "abstract"}]) == []


def test_missing_api_key_short_circuits(monkeypatch):
    monkeypatch.setattr(ls, "_polypizza_headers", lambda: {})
    with patch.object(ls, "_search_polypizza") as search:
        assert ls.fetch_live_assets([{"concept": "dolphin"}]) == []
    search.assert_not_called()


def test_result_without_download_url_skipped(poly_env):
    bad = {"Title": "Dolphin", "Tags": ["sea"]}      # no Download key
    with patch.object(ls, "_search_polypizza", return_value=[bad]):
        assert ls.fetch_live_assets([{"concept": "dolphin", "category": "abstract"}]) == []


def test_blank_concept_skipped(poly_env):
    with patch.object(ls, "_search_polypizza") as search:
        assert ls.fetch_live_assets([{"concept": "", "category": "abstract"}]) == []
    search.assert_not_called()


def test_existing_file_reused_without_download(poly_env):
    """
    The self-heal path after the KB cleanup: a concept whose GLB is already on
    disk must be re-registered from cache, not re-downloaded.
    """
    category_dir = poly_env / "abstract"
    category_dir.mkdir(parents=True, exist_ok=True)
    (category_dir / "basket_ball.glb").write_bytes(b"glb")

    with patch.object(ls, "_search_polypizza", return_value=[_result("Basket Ball", [])]), \
         patch.object(ls, "_download_file") as dl:
        found = ls.fetch_live_assets([{"concept": "basketball", "category": "abstract"}])

    assert len(found) == 1
    assert found[0]["source"] == "cache"
    dl.assert_not_called()
