"""Recency baseline: prefer whatever happened most recently."""

from __future__ import annotations

import math

from contextio.memory import MemoryStore
from contextio.retrieval.base import Query, Scored, rank


def recency_decay(age: float, half_life: float) -> float:
    """Exponential decay in [0, 1]: 1.0 at age 0, 0.5 after one half-life."""
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    return math.exp(-math.log(2) * max(0.0, age) / half_life)


class RecencyRetriever:
    """Scores memories by age.

    With ``half_life`` set, scores decay exponentially with age. Without it, memories are scored
    ``1 / (1 + rank)`` by recency, which reproduces the classic "last N messages" baseline.
    """

    name = "recency"

    def __init__(self, half_life: float | None = None) -> None:
        if half_life is not None and half_life <= 0:
            raise ValueError("half_life must be positive")
        self.half_life = half_life

    def score(self, query: Query, store: MemoryStore) -> list[Scored]:
        if self.half_life is not None:
            now = query.now(store)
            return rank(Scored(m, recency_decay(now - m.timestamp, self.half_life)) for m in store)

        newest_first = sorted(store, key=lambda m: (-m.timestamp, m.id))
        return [Scored(m, 1.0 / (1 + i)) for i, m in enumerate(newest_first)]
