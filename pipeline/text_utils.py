"""
Shared text normalization for the semantic parser and the live-search candidate
builder. Both modules previously kept their own STOPWORDS copy and their own
n-gram loop, which let them drift apart and produce different garbage for the
same input.

Key rule enforced here: an n-gram may never span a removed stopword. Stripping
stopwords from a flat token list and then reading adjacent pairs makes words
that were never adjacent look adjacent -- "saturn and its rings" collapses to
["saturn", "its", "rings"] and yields the phrase "saturn its rings", which then
gets searched verbatim on Poly Pizza (it matched an onion-rings model).
`phrase_runs()` splits at stopwords instead, so n-grams only ever come from
genuinely contiguous words.

Since the 2026-08 pipeline restructure this matters considerably more: live
search now runs on the Stage-2 *optimized* prompt rather than the raw
transcript. The optimizer's job is to produce fuller, more fluent prose, so the
same bug generates roughly 3x more garbage phrases than it used to -- hence
PROSE_FILLER below, which strips the descriptive scaffolding the optimizer
introduces ("a scene showing...", "detailed", "prominent") before any n-gram is
formed.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Stopwords -- single source of truth for the whole pipeline
# ---------------------------------------------------------------------------

STOPWORDS: frozenset[str] = frozenset({
    # articles / conjunctions / prepositions
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "the", "to", "with", "into", "onto",
    "over", "under", "above", "below", "near", "next", "around",
    # pronouns and possessives -- "its" was missing, which is how the phrase
    # "its rings" reached Poly Pizza and matched an onion-rings model
    "its", "his", "her", "hers", "their", "theirs", "our", "ours", "my",
    "mine", "your", "yours", "me", "him", "them", "us", "you", "they",
    # determiners
    "this", "that", "these", "those", "some", "any", "each", "every",
    # imperative verbs that open a request
    "show", "give", "make", "create", "see", "let", "display", "render",
    "build", "generate", "draw", "add",
})


# ---------------------------------------------------------------------------
# Prose scaffolding introduced by the Stage-2 prompt optimizer
# ---------------------------------------------------------------------------
#
# Evidence from the polluted knowledge base -- every one of these concept ids
# was registered from a fragment of optimizer output, not a user command:
#   'scene showing saturn'  'google-style technical office' -> office_chair.glb
#   'create scene fierce'   -> dragon_rigged_1.glb
#   'can deploy this'       -> cargo_depot.glb
#   'need beautiful'        -> make_up_pallete_open.glb
# These words describe *how* to present a scene, never a thing to fetch a model
# of, so they break phrase runs rather than joining them.

PROSE_FILLER: frozenset[str] = frozenset({
    "scene", "scenes", "view", "views", "visualization", "visualisation",
    "showing", "shows", "shown", "showcasing", "featuring", "features",
    "displaying", "displays", "depicting", "depicts", "illustrating",
    "containing", "contains", "including", "includes", "consisting",
    "detailed", "realistic", "beautiful", "prominent", "complete", "full",
    "simple", "basic", "clear", "nice", "good", "great", "proper",
    "style", "styled", "technical", "accurate", "precise", "clean",
    "need", "needs", "want", "wants", "like", "please", "can", "could",
    "would", "should", "will", "deploy", "use", "using", "used",
    "model", "models", "image", "picture", "example", "version",
    # quantifiers and demonstrative verbs -- these produced 'single red',
    # 'single blue', 'blue ball exhibiting', 'bunch flowers' in the KB
    "single", "double", "multiple", "several", "various", "bunch",
    "exhibiting", "exhibit", "demonstrating", "demonstrate", "off",
    # NOTE: deliberately NOT blocking "set", "group", "kind", "type" --
    # "swing set" and "chess set" are real objects with real meshes on disk.
})


# ---------------------------------------------------------------------------
# Motion / state words -- animation, not objects
# ---------------------------------------------------------------------------
#
# These describe motion the physics system already handles from the intent.
# Excluded from asset-search phrases only; the semantic parser still sees them
# so it can infer animation type.

MOTION_WORDS: frozenset[str] = frozenset({
    "bouncing", "bounce", "falling", "fall", "dropping", "drop",
    "swinging", "swing", "rotating", "rotate", "spinning", "spin",
    "orbiting", "orbit", "moving", "move", "floating", "float",
    "flying", "fly", "oscillating", "oscillate", "vibrating", "vibrate",
    "rolling", "roll", "sliding", "slide", "pulsing", "pulse",
    "glowing", "glow", "exploding", "explode", "expanding", "expand",
    "animated", "animate", "simulation", "simulate", "swimming", "rising",
})


# ---------------------------------------------------------------------------
# Concepts that must never be fetched as a downloaded mesh
# ---------------------------------------------------------------------------
#
# scene_architect._RULES already states that astronomical, atomic, molecular and
# abstract-field concepts must be built from primitives. But the architect's
# highest-priority rule says any concept present in AVAILABLE MESHES *must* use
# type="mesh" -- so the moment live search downloads a GLB for "saturn", the two
# rules contradict and the mesh rule wins. The fix belongs here, at retrieval
# time: if a concept is primitive-only, never fetch a mesh for it at all.

PRIMITIVE_ONLY_CONCEPTS: frozenset[str] = frozenset({
    # astronomical
    "star", "sun", "planet", "moon", "ring", "asteroid", "belt", "galaxy",
    "nebula", "comet", "orbit", "solar", "system", "satellite", "cosmos",
    "universe", "constellation", "eclipse", "crater",
    "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus",
    "neptune", "pluto",
    # atomic / subatomic
    "atom", "nucleus", "proton", "neutron", "electron", "shell", "orbital",
    "isotope", "ion", "quark", "particle",
    # molecular / chemical
    "molecule", "bond", "compound", "lattice", "crystal", "polymer",
    # abstract fields and physics
    "wave", "field", "energy", "force", "vector", "magnetic", "electric",
    "gravity", "gravitational", "photon", "quantum", "plasma", "spectrum",
    "frequency", "amplitude", "oscillation", "trajectory",
})


# Everything that should break a phrase run when building asset-search queries.
SEARCH_STOP: frozenset[str] = STOPWORDS | PROSE_FILLER | MOTION_WORDS


def is_primitive_only(concept: str) -> bool:
    """
    True if the concept must be rendered from primitives, so live search should
    skip it. Matches if *any* word in the phrase is primitive-only -- "saturn
    rings" and "solar system" both qualify.
    """
    words = tokenize(concept)
    if not words:
        return False
    return any(singularize(w) in PRIMITIVE_ONLY_CONCEPTS for w in words)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphabetic tokens, in order. No stopword filtering."""
    return _WORD_RE.findall((text or "").lower())


# Endings that look plural but are not. Stripping the trailing "s" from these
# broke the primitive-only gate: "nucleus" became "nucleu", which is not in
# PRIMITIVE_ONLY_CONCEPTS, so a carbon-atom prompt downloaded a mesh for its
# nucleus. "torus" and "radius" are the same shape of bug and both name things
# the schema uses.
_NOT_PLURAL_ENDINGS = ("ss", "us", "is", "as", "os", "ries")


def singularize(word: str) -> str:
    """
    Crude plural stripper, good enough for matching asset titles against
    queries ("rings" vs "ring", "planets" vs "planet").

    Deliberately conservative: when unsure it returns the word unchanged.
    A missed plural only costs a fuzzy match; an over-eager strip silently
    breaks concept-set membership, which is the more expensive failure.
    """
    if len(word) <= 3 or word.endswith(_NOT_PLURAL_ENDINGS):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"              # bodies -> body
    if word.endswith(("xes", "ses", "shes", "ches", "zes")):
        return word[:-2]                    # boxes -> box, dishes -> dish
    if word.endswith("s"):
        return word[:-1]                    # rings -> ring, eyes -> eye
    return word


def phrase_runs(
    text: str,
    min_len: int = 3,
    extra_stop: frozenset[str] | set[str] | None = None,
) -> list[list[str]]:
    """
    Split `text` into runs of consecutive content words, breaking at every
    stopword or short word.

        "solar system with saturn and its rings"
          -> [["solar", "system"], ["saturn"], ["rings"]]

    "saturn" and "rings" land in separate runs because "and its" sits between
    them, so no n-gram can ever join them.
    """
    skip = STOPWORDS | (extra_stop or frozenset())
    runs: list[list[str]] = []
    current: list[str] = []

    for token in tokenize(text):
        if token in skip or len(token) < min_len:
            if current:
                runs.append(current)
                current = []
            continue
        current.append(token)

    if current:
        runs.append(current)
    return runs


def contiguous_ngrams(
    text: str,
    sizes: tuple[int, ...] = (3, 2),
    min_len: int = 3,
    extra_stop: frozenset[str] | set[str] | None = None,
) -> list[str]:
    """
    All n-grams of the given sizes that lie wholly inside a single run of
    contiguous content words. Longest sizes first, deduplicated, order stable.
    """
    phrases: list[str] = []
    for run in phrase_runs(text, min_len=min_len, extra_stop=extra_stop):
        for n in sizes:
            if n > len(run):
                continue
            for i in range(len(run) - n + 1):
                phrase = " ".join(run[i : i + n])
                if phrase not in phrases:
                    phrases.append(phrase)
    return phrases


# ---------------------------------------------------------------------------
# Live-search result relevance
# ---------------------------------------------------------------------------

def result_matches_query(query: str, title: str, tags: list[str] | None = None) -> bool:
    """
    True if a Poly Pizza result plausibly answers the query.

    The API returns a best-effort ranking and never says "no match", so without
    this check the top hit is accepted no matter how unrelated -- searching
    "saturn" returned a generic "Planets" model. Requires at least one shared
    content word between the query and the result's title or tags.
    """
    q_tokens = [w for w in tokenize(query) if w not in STOPWORDS and len(w) >= 3]
    if not q_tokens:
        return False

    q_words = {singularize(w) for w in q_tokens}
    candidate_words = {singularize(w) for w in tokenize(title)}
    for tag in tags or []:
        candidate_words.update(singularize(w) for w in tokenize(str(tag)))

    if q_words & candidate_words:
        return True

    # Compound-word fallback: "basketball" and a model titled "Basket Ball"
    # share no token but are the same thing. Compare with word breaks removed.
    q_joined = "".join(q_tokens)
    title_joined = "".join(tokenize(title))
    if len(q_joined) >= 5 and len(title_joined) >= 5:
        if q_joined in title_joined or title_joined in q_joined:
            return True

    return False
