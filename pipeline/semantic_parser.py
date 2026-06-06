"""
Semantic intent parser using sentence-transformer embeddings.

The concept corpus is loaded from MongoDB. Instantiate SemanticParser once at
startup; model loading and corpus encoding are intentionally not repeated per
command.
"""

from __future__ import annotations

import re
import threading

import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.knowledge_base.mongo_client import get_all_concepts

SIMILARITY_THRESHOLD = 0.35
OBJECT_EMBEDDING_THRESHOLD = 0.40
MODEL_NAME = "all-MiniLM-L6-v2"

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "show",
    "the",
    "to",
    "with",
    "you",
    "your",
}


class SemanticParser:
    """Load model + concept corpus once, then parse many transcripts quickly."""

    def __init__(self, threshold: float = SIMILARITY_THRESHOLD) -> None:
        self.threshold = threshold
        print("[semantic_parser] Loading sentence-transformer model...")
        # low_cpu_mem_usage=False avoids meta-tensor initialization that breaks
        # on PyTorch ≥ 2.7 where .to() on meta tensors is no longer allowed.
        self._model = SentenceTransformer(
            MODEL_NAME,
            model_kwargs={"low_cpu_mem_usage": False},
        )

        docs = get_all_concepts()
        if not docs:
            raise RuntimeError("No concepts found in MongoDB knowledge base.")

        self._concept_map: dict = {}
        self._descriptions: dict = {}
        self._synonym_map: dict[str, list[str]] = {}
        for doc in docs:
            concept = str(doc.get("_id") or "").lower().strip()
            if not concept:
                continue
            self._concept_map[concept] = {
                "type": doc.get("type", "object"),
                "asset_id": doc.get("asset_id"),
                "asset_src": doc.get("asset_src"),
                "category": doc.get("category"),
            }
            synonyms = [
                str(alias).lower().strip()
                for alias in doc.get("synonyms", [])
                if isinstance(alias, str) and alias.strip()
            ]
            description_parts = [
                concept,
                str(doc.get("description") or ""),
                str(doc.get("category") or ""),
                *synonyms,
            ]
            self._descriptions[concept] = " ".join(
                part for part in description_parts if part.strip()
            )
            for alias in synonyms:
                self._synonym_map.setdefault(alias, [])
                if concept not in self._synonym_map[alias]:
                    self._synonym_map[alias].append(concept)
        self._phrase_candidates: list[tuple[list[str], str]] = self._build_phrase_candidates()
        self._concepts: list[str] = list(self._descriptions.keys())
        corpus = [self._descriptions[concept] for concept in self._concepts]

        print(f"[semantic_parser] Encoding {len(corpus)} concept descriptions...")
        self._corpus_matrix: np.ndarray = self._model.encode(
            corpus,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        print("[semantic_parser] Ready.")

    def _scores(self, text: str) -> np.ndarray:
        query_vec: np.ndarray = self._model.encode(
            text.lower().strip(),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return self._corpus_matrix @ query_vec

    def parse_intent(self, text: str) -> dict:
        if not text or not text.strip():
            return {"objects": [], "structures": [], "systems": [], "effects": []}

        phrase_hits, phrase_tokens = self._match_phrases(text)
        blocked_tokens = set(phrase_tokens)

        buckets = {"objects": [], "structures": [], "systems": [], "effects": []}
        type_to_bucket = {
            "object": "objects",
            "structure": "structures",
            "system": "systems",
            "effect": "effects",
        }

        for concept in phrase_hits:
            entry = self._concept_map.get(concept)
            if not entry:
                continue
            bucket = type_to_bucket.get(entry.get("type"))
            if bucket and concept not in buckets[bucket]:
                buckets[bucket].append(concept)

        # Apply lexical matches before embeddings to lock explicit object hits.
        self._apply_lexical_supplement(text, buckets, type_to_bucket, blocked_tokens)
        locked_objects = set(buckets["objects"])

        scores = self._scores(text)

        scored_concepts = sorted(
            ((concept, float(scores[idx])) for idx, concept in enumerate(self._concepts)),
            key=lambda item: item[1],
            reverse=True,
        )
        seen_embedding_categories: set[str] = set()

        for concept, score in scored_concepts:
            if score < self.threshold:
                break
            entry = self._concept_map.get(concept)
            if not entry:
                continue
            if concept in blocked_tokens and concept not in phrase_hits:
                continue
            bucket = type_to_bucket.get(entry.get("type"))
            if bucket == "objects" and locked_objects and concept not in locked_objects:
                continue
            if bucket == "objects" and score < OBJECT_EMBEDDING_THRESHOLD:
                continue
            if bucket == "objects":
                category = str(entry.get("category") or "")
                if category in seen_embedding_categories:
                    continue
                if category:
                    seen_embedding_categories.add(category)
            if bucket and concept not in buckets[bucket]:
                buckets[bucket].append(concept)

        buckets["_phrase_hits"] = phrase_hits
        buckets["_unresolved_tokens"] = self._find_unresolved_tokens(text, blocked_tokens)
        return buckets

    def _apply_lexical_supplement(
        self,
        text: str,
        buckets: dict[str, list[str]],
        type_to_bucket: dict[str, str],
        blocked_tokens: set[str],
    ) -> None:
        words = re.findall(r"[a-zA-Z][a-zA-Z_-]*", text.lower())
        candidates: list[str] = []
        for word in words:
            if word in blocked_tokens:
                continue
            candidates.append(word)
            if word.endswith("s") and len(word) > 3:
                candidates.append(word[:-1])
            if word.endswith("ing") and len(word) > 5:
                candidates.append(word[:-3])

        for token in candidates:
            targets: list[str] = []
            synonym_targets = [
                synonym_target
                for synonym_target in self._synonym_map.get(token, [])
                if isinstance(synonym_target, str)
            ]
            if synonym_targets:
                targets.extend(synonym_targets)
            elif token.endswith("s") and token[:-1] in self._concept_map:
                targets.append(token[:-1])
            elif token in self._concept_map:
                targets.append(token)

            for target in targets:
                entry = self._concept_map.get(target)
                if not entry:
                    continue
                bucket = type_to_bucket.get(entry.get("type"))
                if bucket and target not in buckets[bucket]:
                    buckets[bucket].append(target)

    def _normalize_phrase(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z][a-zA-Z_-]*", text.lower())

    def _build_phrase_candidates(self) -> list[tuple[list[str], str]]:
        candidates: list[tuple[list[str], str]] = []

        for concept in self._concept_map.keys():
            tokens = self._normalize_phrase(concept.replace("_", " ").replace("-", " "))
            if len(tokens) >= 2:
                candidates.append((tokens, concept))

        for phrase, targets in self._synonym_map.items():
            if not isinstance(phrase, str) or " " not in phrase:
                continue
            tokens = self._normalize_phrase(phrase)
            if len(tokens) < 2:
                continue
            for target in targets:
                if target in self._concept_map:
                    candidates.append((tokens, target))

        candidates.sort(key=lambda item: len(item[0]), reverse=True)
        return candidates

    def _match_phrases(self, text: str) -> tuple[list[str], list[str]]:
        tokens = self._normalize_phrase(text)
        if not tokens:
            return [], []

        used = [False] * len(tokens)
        hits: list[str] = []
        hit_tokens: list[str] = []

        for phrase_tokens, concept in self._phrase_candidates:
            plen = len(phrase_tokens)
            if plen == 0 or plen > len(tokens):
                continue
            for i in range(0, len(tokens) - plen + 1):
                if any(used[i : i + plen]):
                    continue
                if tokens[i : i + plen] == phrase_tokens:
                    hits.append(concept)
                    for j in range(i, i + plen):
                        used[j] = True
                        hit_tokens.append(tokens[j])

        return hits, hit_tokens

    def _find_unresolved_tokens(self, text: str, blocked_tokens: set[str]) -> list[str]:
        tokens = self._normalize_phrase(text)
        unresolved: list[str] = []
        clean_tokens = [t for t in tokens if t not in _STOPWORDS and t not in blocked_tokens]
        for token in clean_tokens:
            if token in self._concept_map:
                continue
            if token in self._synonym_map:
                continue
            if token not in unresolved:
                unresolved.append(token)

        # Include short noun-phrase candidates to prefer phrase-level live search.
        phrase_tokens: set[str] = set()
        for size in (3, 2):
            for i in range(0, len(clean_tokens) - size + 1):
                phrase = " ".join(clean_tokens[i : i + size])
                if phrase in self._concept_map or phrase in self._synonym_map:
                    continue
                if phrase not in unresolved:
                    unresolved.append(phrase)
                phrase_tokens.update(clean_tokens[i : i + size])

        if phrase_tokens:
            unresolved = [item for item in unresolved if item not in phrase_tokens]

        return unresolved

    def top_matches(self, text: str, n: int = 10) -> list[tuple[str, float]]:
        if not text or not text.strip():
            return []
        scores = self._scores(text)
        top_idx = np.argsort(scores)[::-1][:n]
        return [(self._concepts[idx], float(scores[idx])) for idx in top_idx]


_parser_instance: SemanticParser | None = None
_parser_lock = threading.Lock()


def get_parser() -> SemanticParser:
    global _parser_instance
    if _parser_instance is not None:
        return _parser_instance
    with _parser_lock:
        if _parser_instance is None:
            _parser_instance = SemanticParser()
    return _parser_instance


def parse_intent(text: str) -> dict:
    return get_parser().parse_intent(text)


if __name__ == "__main__":
    import sys

    prompt = " ".join(sys.argv[1:]) or "red rocky world with orbiting satellites"
    parser = SemanticParser()

    print(f"\nTranscript: {prompt!r}")
    print("\nTop 10 concept matches:")
    for concept, score in parser.top_matches(prompt, 10):
        bar = "#" * max(0, int(score * 30))
        print(f"  {score:.3f} {bar:<30} {concept}")

    print("\nParsed intent:")
    print(json.dumps(parser.parse_intent(prompt), indent=2))
