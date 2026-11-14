"""
pipeline/kb_cleanup.py — classification safety.

This script deletes from a MongoDB Atlas M0 cluster, which has no automated
backups. The single most important property is that it can NEVER select a seed
concept: seeds are the 216 curated entries from asset_ingester.py, and losing
them is unrecoverable (run_ingestion() re-seeds only ~63 generic assets from a
hardcoded query list, not the accumulated library).
"""
from __future__ import annotations

import json

import pytest

from pipeline.kb_cleanup import _MANUAL_JUNK_IDS, backup, classify


# ── the core safety guarantee ────────────────────────────────────────────────

@pytest.mark.parametrize(
    "concept_id",
    ["human figure", "palm tree", "police car", "human heart",
     "office building", "minor planet", "mermaid swimming"],
)
def test_seed_concepts_are_never_deleted(concept_id, make_concept_doc):
    """
    Seed concepts are protected by provenance alone. Several are multi-word and
    would otherwise look like fragments — 'mermaid swimming' even contains a
    motion verb — so only the description marker keeps them safe.
    """
    doc = make_concept_doc(concept_id, "/assets/meshes/abstract/brain.glb",
                           live_search=False)
    buckets = classify([doc])
    assert buckets["keep"] == [doc]
    assert buckets["fragment"] == []
    assert buckets["manual"] == []


def test_seed_concept_matching_a_manual_junk_id_still_protected(make_concept_doc):
    """Provenance beats the manual list — belt and braces."""
    doc = make_concept_doc("harmonic motion", "/assets/meshes/abstract/brain.glb",
                           live_search=False)
    assert classify([doc])["keep"] == [doc]


# ── fragment detection ───────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "concept_id",
    ["its rings", "falling hardwood", "can deploy this", "need beautiful",
     "google-style technical office", "single red", "blue ball exhibiting",
     "off surface", "scene depicting our"],
)
def test_live_search_fragments_selected(concept_id, make_concept_doc):
    doc = make_concept_doc(concept_id, "/assets/meshes/abstract/brain.glb")
    assert classify([doc])["fragment"] == [doc]


@pytest.mark.parametrize(
    "concept_id", ["brain", "dolphin", "velociraptor", "sunflower", "kidney"])
def test_clean_live_search_concepts_kept(concept_id, make_concept_doc):
    doc = make_concept_doc(concept_id, "/assets/meshes/abstract/brain.glb")
    assert classify([doc])["keep"] == [doc]


def test_manual_junk_ids_selected(make_concept_doc):
    doc = make_concept_doc("harmonic motion", "/assets/meshes/abstract/brain.glb")
    assert classify([doc])["manual"] == [doc]


def test_every_manual_id_has_a_documented_reason():
    for concept_id, reason in _MANUAL_JUNK_IDS.items():
        assert reason.strip(), f"{concept_id} has no documented reason"


# ── broken links ─────────────────────────────────────────────────────────────

def test_missing_asset_file_goes_to_broken_link(make_concept_doc):
    doc = make_concept_doc("character", "/assets/meshes/abstract/not_here.glb",
                           live_search=False)
    assert classify([doc])["broken_link"] == [doc]


def test_empty_asset_src_is_not_a_broken_link(make_concept_doc):
    """effect/system/structure concepts legitimately carry no mesh."""
    for doc_type in ("effect", "system", "structure"):
        doc = make_concept_doc("orbit", "", live_search=False, doc_type=doc_type)
        buckets = classify([doc])
        assert buckets["broken_link"] == []
        assert buckets["keep"] == [doc]


def test_broken_link_takes_priority_over_fragment(make_concept_doc):
    """A missing file is reported for review, never silently deleted."""
    doc = make_concept_doc("its rings", "/assets/meshes/abstract/gone.glb")
    buckets = classify([doc])
    assert buckets["broken_link"] == [doc]
    assert buckets["fragment"] == []


# ── bucket integrity ─────────────────────────────────────────────────────────

def test_every_document_lands_in_exactly_one_bucket(make_concept_doc):
    docs = [
        make_concept_doc("its rings", "/assets/meshes/abstract/brain.glb"),
        make_concept_doc("harmonic motion", "/assets/meshes/abstract/brain.glb"),
        make_concept_doc("brain", "/assets/meshes/abstract/brain.glb"),
        make_concept_doc("palm tree", "/assets/meshes/abstract/brain.glb", live_search=False),
        make_concept_doc("gone", "/assets/meshes/abstract/missing.glb"),
    ]
    buckets = classify(docs)
    total = sum(len(v) for v in buckets.values())
    assert total == len(docs)
    ids = [d["_id"] for bucket in buckets.values() for d in bucket]
    assert len(ids) == len(set(ids))


def test_classify_on_empty_input():
    buckets = classify([])
    assert all(v == [] for v in buckets.values())


# ── backup round-trip (the only route back from a bad delete) ────────────────

def test_backup_writes_restorable_json(tmp_path, monkeypatch, make_concept_doc):
    import pipeline.kb_cleanup as mod
    monkeypatch.setattr(mod, "_BACKUP_DIR", tmp_path)

    docs = [make_concept_doc("its rings", "/assets/meshes/abstract/onion_rings.glb")]
    path = backup(docs)

    assert path.exists()
    restored = json.loads(path.read_text(encoding="utf-8"))
    assert restored == docs
    assert restored[0]["_id"] == "its rings"
