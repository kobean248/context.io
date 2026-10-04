"""Importance-weighted hybrid scoring.

    S(m, q) = w_r * R(m) + w_s * Sim(m, q) + w_i * I(m) + w_f * F(m)

where R is recency decay, Sim is query similarity fused from one or more retrievers, I is the
memory's importance, and F is how often the memory has been used before.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from contextio.memory import MemoryKind, MemoryStore
from contextio.retrieval.base import Query, Retriever, Scored, rank
from contextio.retrieval.lexical import BM25Retriever
from contextio.retrieval.recency import recency_decay
from contextio.retrieval.vector import VectorRetriever


@dataclass(frozen=True)
class HybridWeights:
    recency: float = 0.15
    similarity: float = 0.6
    importance: float = 0.15
    frequency: float = 0.1

    def __post_init__(self) -> None:
        if min(self.recency, self.similarity, self.importance, self.frequency) < 0:
            raise ValueError("weights must be non-negative")


def normalize_scores(scored: Sequence[Scored]) -> dict[str, float]:
    """Map scores into [0, 1] by dividing by the maximum; negative scores become 0."""
    top = max((s.score for s in scored), default=0.0)
    if top <= 0:
        return {s.memory.id: 0.0 for s in scored}
    return {s.memory.id: max(0.0, s.score) / top for s in scored}


class HybridRetriever:
    """Combine similarity, recency, importance, and usage frequency into one score.

    Args:
        similarity: retrievers whose normalized scores are averaged into Sim. Defaults to BM25
            plus vector retrieval.
        weights: relative weight of each signal.
        half_life: recency half-life, in the same units as memory timestamps.
        min_similarity: memories whose normalized similarity is below this threshold score 0,
            so importance or recency alone cannot pull an off-topic memory into the context.
        pin_static: let static memories bypass the similarity gate. Static profile facts are
            short and stable, so always including them is cheap and cache-friendly.
    """

    name = "hybrid"

    def __init__(
        self,
        similarity: Retriever | Sequence[Retriever] | None = None,
        weights: HybridWeights | None = None,
        half_life: float = 30.0,
        min_similarity: float = 0.1,
        pin_static: bool = False,
    ) -> None:
        if similarity is None:
            similarity = (BM25Retriever(), VectorRetriever())
        self.retrievers: tuple[Retriever, ...] = (
            (similarity,) if isinstance(similarity, Retriever) else tuple(similarity)
        )
        if not self.retrievers:
            raise ValueError("at least one similarity retriever is required")
        self.weights = weights or HybridWeights()
        self.half_life = half_life
        self.min_similarity = min_similarity
        self.pin_static = pin_static

    def similarity(self, query: Query, store: MemoryStore) -> dict[str, float]:
        per_retriever = [normalize_scores(r.score(query, store)) for r in self.retrievers]
        return {
            m.id: sum(scores.get(m.id, 0.0) for scores in per_retriever) / len(per_retriever)
            for m in store
        }

    def score(self, query: Query, store: MemoryStore) -> list[Scored]:
        w = self.weights
        now = query.now(store)
        similarity = self.similarity(query, store)
        max_usage = store.max_usage

        results = []
        for memory in store:
            sim = similarity[memory.id]
            is_static = memory.kind is MemoryKind.STATIC
            if sim < self.min_similarity and not (self.pin_static and is_static):
                results.append(Scored(memory, 0.0))
                continue
            recency = 1.0 if is_static else recency_decay(now - memory.timestamp, self.half_life)
            frequency = (
                math.log1p(store.usage(memory.id)) / math.log1p(max_usage) if max_usage else 0.0
            )
            total = (
                w.recency * recency
                + w.similarity * sim
                + w.importance * memory.importance
                + w.frequency * frequency
            )
            results.append(Scored(memory, total))
        return rank(results)
