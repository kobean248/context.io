"""Dense vector retrieval with a pluggable embedder."""

from __future__ import annotations

import math
import operator
import zlib
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol, runtime_checkable

from contextio.memory import MemoryStore
from contextio.retrieval.base import Query, Scored, StoreCache, rank
from contextio.text import char_ngrams, tokenize

Vector = Sequence[float]


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[Vector]: ...


class HashingEmbedder:
    """Deterministic, dependency-free embedder based on the hashing trick.

    Words, word bigrams, and character n-grams are hashed into a fixed number of signed buckets
    with sublinear term-frequency weighting. Character n-grams give some robustness to
    morphology ("deploy" vs "deployment"). It has no notion of synonyms, which makes it a fair
    stand-in for a weak embedding model; swap in a real ``Embedder`` for stronger baselines.
    """

    def __init__(
        self,
        dim: int = 512,
        ngram: int = 3,
        word_weight: float = 1.0,
        bigram_weight: float = 0.5,
        char_weight: float = 0.3,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.ngram = ngram
        self.word_weight = word_weight
        self.bigram_weight = bigram_weight
        self.char_weight = char_weight

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._embed_one(text) for text in texts]

    def _features(self, text: str) -> Counter[str]:
        terms = tokenize(text)
        features: Counter[str] = Counter()
        for term in terms:
            features[f"w:{term}"] += 1
            if self.char_weight:
                for gram in char_ngrams(term, self.ngram):
                    features[f"c:{gram}"] += 1
        if self.bigram_weight:
            for left, right in pairwise(terms):
                features[f"b:{left}_{right}"] += 1
        return features

    def _embed_one(self, text: str) -> list[float]:
        weights = {"w": self.word_weight, "b": self.bigram_weight, "c": self.char_weight}
        vector = [0.0] * self.dim
        for feature, count in self._features(text).items():
            digest = zlib.crc32(feature.encode())
            sign = 1.0 if digest & 1 else -1.0
            vector[(digest >> 1) % self.dim] += sign * weights[feature[0]] * (1 + math.log(count))
        return _normalize(vector)


def _normalize(vector: Vector) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else list(vector)


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity of two already L2-normalized vectors."""
    return sum(map(operator.mul, a, b))


class VectorRetriever:
    """Cosine similarity between the query and memory embeddings.

    Memory embeddings are computed once per store and cached; only the query is embedded per
    request.
    """

    name = "vector"

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self._cache: StoreCache[dict[str, list[float]]] = StoreCache(self._build)

    def _build(self, store: MemoryStore) -> dict[str, list[float]]:
        memories = list(store)
        vectors = self.embedder.embed([m.text for m in memories]) if memories else []
        return {m.id: _normalize(v) for m, v in zip(memories, vectors, strict=True)}

    def score(self, query: Query, store: MemoryStore) -> list[Scored]:
        index = self._cache.get(store)
        if not index:
            return []
        query_vector = _normalize(self.embedder.embed([query.text])[0])
        return rank(Scored(m, cosine(query_vector, index[m.id])) for m in store)
