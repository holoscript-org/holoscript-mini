"""
Semantic intent parser using sentence-transformer embeddings.

This replaces brittle keyword matching with cosine similarity over the generated
concept_descriptions.json corpus. Instantiate SemanticParser once at startup;
model loading and corpus encoding are intentionally not repeated per command.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

_KB = Path(__file__).parent / "knowledge_base"
_MAP = _KB / "concept_map.json"
_DESC = _KB / "concept_descriptions.json"
_SYN = _KB / "synonym_map.json"

SIMILARITY_THRESHOLD = 0.35
OBJECT_EMBEDDING_THRESHOLD = 0.40
MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticParser:
    """Load model + concept corpus once, then parse many transcripts quickly."""

    def __init__(self, threshold: float = SIMILARITY_THRESHOLD) -> None:
        if not _MAP.exists():
            raise RuntimeError("concept_map.json not found. Run: python -m pipeline.kb_builder")
        if not _DESC.exists():
            raise RuntimeError(
                "concept_descriptions.json not found. Run: python -m pipeline.kb_builder"
            )

        self.threshold = threshold
        print("[semantic_parser] Loading sentence-transformer model...")
        self._model = SentenceTransformer(MODEL_NAME)

        self._concept_map: dict = json.loads(_MAP.read_text(encoding="utf-8"))
        self._descriptions: dict = json.loads(_DESC.read_text(encoding="utf-8"))
        self._synonym_map: dict = {}
        if _SYN.exists():
            raw_syn = json.loads(_SYN.read_text(encoding="utf-8"))
            self._synonym_map = {
                key: value
                for key, value in raw_syn.items()
                if isinstance(key, str) and not key.startswith("_") and isinstance(value, list)
            }
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

        scores = self._scores(text)
        buckets = {"objects": [], "structures": [], "systems": [], "effects": []}
        type_to_bucket = {
            "object": "objects",
            "structure": "structures",
            "system": "systems",
            "effect": "effects",
        }

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
            bucket = type_to_bucket.get(entry.get("type"))
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

        self._apply_lexical_supplement(text, buckets, type_to_bucket)
        return buckets

    def _apply_lexical_supplement(
        self,
        text: str,
        buckets: dict[str, list[str]],
        type_to_bucket: dict[str, str],
    ) -> None:
        words = re.findall(r"[a-zA-Z][a-zA-Z_-]*", text.lower())
        candidates: list[str] = []
        for word in words:
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

    def top_matches(self, text: str, n: int = 10) -> list[tuple[str, float]]:
        if not text or not text.strip():
            return []
        scores = self._scores(text)
        top_idx = np.argsort(scores)[::-1][:n]
        return [(self._concepts[idx], float(scores[idx])) for idx in top_idx]


_parser_instance: SemanticParser | None = None


def get_parser() -> SemanticParser:
    global _parser_instance
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
