"""
pipeline/text_utils.py — the phrase-extraction and relevance-gating rules.

These tests exist because a regression here silently poisons the MongoDB
knowledge base: live_search registers whatever query string it was given as a
permanent concept `_id`. 41 junk documents had to be deleted on 2026-08-05
because these rules were wrong. Each test below names the real failure it
guards against.
"""
from __future__ import annotations

import pytest

from pipeline.text_utils import (
    MOTION_WORDS,
    PRIMITIVE_ONLY_CONCEPTS,
    PROSE_FILLER,
    SEARCH_STOP,
    STOPWORDS,
    contiguous_ngrams,
    is_primitive_only,
    phrase_runs,
    result_matches_query,
    singularize,
    tokenize,
)


# ── tokenize / singularize ───────────────────────────────────────────────────

def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("A Google-Style Office, showing 3 desks!") == [
        "a", "google", "style", "office", "showing", "desks",
    ]


def test_tokenize_handles_empty_and_none():
    assert tokenize("") == []
    assert tokenize(None) == []  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "word,expected",
    [
        ("rings", "ring"),
        ("planets", "planet"),
        ("boxes", "box"),
        ("glass", "glass"),    # -ss must not be stripped
        ("gas", "gas"),        # too short to strip
        ("series", "series"),  # -ies left alone
    ],
)
def test_singularize(word, expected):
    assert singularize(word) == expected


# ── the core bug: n-grams must never span a removed stopword ─────────────────

def test_phrase_runs_split_at_stopwords():
    """
    THE regression guard. "saturn and its rings" must not become the phrase
    "saturn its rings" — that query matched an onion-rings model on Poly Pizza
    and was registered as a permanent concept.
    """
    assert phrase_runs("solar system with saturn and its rings") == [
        ["solar", "system"], ["saturn"], ["rings"],
    ]


def test_its_is_a_stopword():
    """The single missing word that started the whole incident."""
    assert "its" in STOPWORDS


def test_possessives_and_pronouns_are_stopwords():
    for word in ["its", "his", "her", "their", "our", "my", "your", "them"]:
        assert word in STOPWORDS, f"{word!r} should be a stopword"


def test_glued_phrase_is_never_produced():
    ngrams = contiguous_ngrams("solar system with saturn and its rings")
    assert "saturn its rings" not in ngrams
    assert "its rings" not in ngrams
    assert ngrams == ["solar system"]


def test_contiguous_ngrams_preserves_real_compounds():
    """Splitting must not be so aggressive that real objects are lost."""
    assert "wrist watch" in contiguous_ngrams("show me a wrist watch on a table")
    assert "grandfather clock" in contiguous_ngrams("a grandfather clock in a room")


def test_ngrams_are_deduplicated_and_ordered_longest_first():
    out = contiguous_ngrams("alpha beta gamma")
    assert out == ["alpha beta gamma", "alpha beta", "beta gamma"]
    assert len(out) == len(set(out))


def test_short_words_break_runs():
    # "of" is a stopword, "ox" is under min_len — both must split the run
    assert phrase_runs("red ox cart") == [["red"], ["cart"]]


# ── prose filler (Stage-2 optimizer output) ──────────────────────────────────

def test_prose_filler_strips_optimizer_scaffolding():
    """
    Live search runs on the *optimized* prompt, which is fluent LLM prose.
    'google-style technical office' and 'create scene fierce' were real
    concept ids created this way.
    """
    # contiguous_ngrams only emits multi-word phrases; every filler word is
    # stripped so no run is long enough to form one. Single surviving words
    # like "desk" reach the search via the separate fallback loop in
    # pipeline_runner._extract_candidate_phrases (covered in its own test).
    out = contiguous_ngrams(
        "A google-style technical office scene showing a desk",
        extra_stop=SEARCH_STOP,
    )
    assert out == []
    for junk in ["google style technical", "technical office scene",
                 "office scene showing", "scene showing"]:
        assert junk not in out


def test_motion_words_excluded_from_search_phrases():
    """Motion is handled by the physics system, never fetched as a model."""
    out = contiguous_ngrams(
        "A basketball falling and bouncing on a hardwood floor",
        extra_stop=SEARCH_STOP,
    )
    assert "basketball falling" not in out
    assert "hardwood floor" in out


def test_search_stop_is_the_union_of_all_three_lists():
    assert SEARCH_STOP == STOPWORDS | PROSE_FILLER | MOTION_WORDS


def test_swing_set_words_deliberately_not_blocked():
    """
    Regression guard for an over-blocking mistake: swing_set.glb and chess sets
    are real assets, so 'set'/'group'/'kind'/'type' must stay searchable even
    though they look like filler.
    """
    for word in ["set", "group", "kind", "type"]:
        assert word not in SEARCH_STOP, f"{word!r} must remain searchable"


# ── primitive-only gate ──────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "concept",
    ["saturn", "rings", "solar system", "its rings", "atom", "electron",
     "water molecule", "magnetic field"],
)
def test_primitive_only_concepts_blocked(concept):
    """
    The architect is told astronomical/atomic concepts are primitives, but ALSO
    that anything in AVAILABLE MESHES must be a mesh — and the mesh rule wins.
    So these must never be fetched in the first place.
    """
    assert is_primitive_only(concept) is True


@pytest.mark.parametrize(
    "concept",
    ["wrist watch", "basketball", "human skeleton", "grandfather clock",
     "wooden floor", "office chair"],
)
def test_real_objects_not_blocked(concept):
    assert is_primitive_only(concept) is False


def test_primitive_only_matches_any_word_in_phrase():
    assert is_primitive_only("shiny saturn model") is True


def test_primitive_only_handles_plurals_and_empty():
    assert is_primitive_only("planets") is True
    assert is_primitive_only("") is False


def test_primitive_only_set_covers_all_planets():
    for planet in ["mercury", "venus", "earth", "mars", "jupiter",
                   "saturn", "uranus", "neptune", "pluto"]:
        assert planet in PRIMITIVE_ONLY_CONCEPTS


# ── Poly Pizza result relevance ──────────────────────────────────────────────

@pytest.mark.parametrize(
    "query,title,tags,expected",
    [
        # rejections — the API always ranks *something* first, never says "no match"
        ("saturn",          "Planets",        ["space"],     False),
        ("harmonic motion", "Harp",           ["music"],     False),
        ("brain",           "Cauliflower",    ["vegetable"], False),
        # direct token overlap
        ("wrist watch",     "Wristwatch",     ["time"],      True),
        ("hardwood floor",  "Wood Floor",     ["floor"],     True),
        ("skeleton",        "Human Skeleton", ["anatomy"],   True),
        # tags rescue a title that shares nothing
        ("dragon",          "Fantasy Beast",  ["dragon"],    True),
        # compound-word fallback
        ("basketball",      "Basket Ball",    ["sport"],     True),
    ],
)
def test_result_matches_query(query, title, tags, expected):
    assert result_matches_query(query, title, tags) is expected


def test_result_match_rejects_query_of_only_stopwords():
    assert result_matches_query("of the a", "Anything", []) is False


def test_result_match_survives_missing_tags():
    assert result_matches_query("dolphin", "Dolphin", None) is True


def test_compound_fallback_requires_min_length():
    """Short joins must not match by accident ('ox' inside 'box')."""
    assert result_matches_query("ox", "Box", []) is False


# ── Latin -us / -is singulars (regression: 2026-08-06) ───────────────────────

@pytest.mark.parametrize(
    "word", ["nucleus", "torus", "radius", "focus", "axis", "basis", "iris"])
def test_latin_singulars_not_stripped(word):
    """
    `singularize` used to strip the trailing s from any non-ss word, turning
    "nucleus" into "nucleu" — which is not in PRIMITIVE_ONLY_CONCEPTS, so a
    carbon-atom prompt downloaded a mesh for its nucleus. "torus" and "radius"
    are the same bug and both name things the schema itself uses.
    """
    assert singularize(word) == word


@pytest.mark.parametrize(
    "plural,expected",
    [("rings", "ring"), ("planets", "planet"), ("electrons", "electron"),
     ("trees", "tree"), ("eyes", "eye"), ("boxes", "box"),
     ("dishes", "dish"), ("bodies", "body")],
)
def test_real_plurals_still_reduced(plural, expected):
    assert singularize(plural) == expected


@pytest.mark.parametrize(
    "concept", ["nucleus", "central nucleus", "atomic nucleus", "the nucleus"])
def test_nucleus_is_primitive_only(concept):
    """The exact leak that put core.glb at the centre of a carbon atom."""
    assert is_primitive_only(concept) is True
