"""Budget-constrained context selection.

Given memories scored by a retriever, a selector chooses the subset that goes into the prompt
without exceeding a token budget. This is a 0/1 knapsack problem: each memory has a weight (its
token count) and a value (its relevance score).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from contextio.memory import Memory, MemoryKind
from contextio.retrieval.base import Scored

TokenFn = Callable[[Memory], int]
Order = Literal["score", "chronological", "static_first"]

_KIND_ORDER = {MemoryKind.STATIC: 0, MemoryKind.SEMANTIC: 1, MemoryKind.EPISODIC: 2}


@dataclass(frozen=True)
class Selection:
    memories: tuple[Memory, ...]
    """Selected memories, in the order the selector chose them."""
    tokens: int
    budget: int | None = None

    @property
    def ids(self) -> list[str]:
        return [m.id for m in self.memories]

    def ordered(self, order: Order = "static_first") -> list[Memory]:
        """Memories arranged for the prompt.

        ``static_first`` puts static memories first in a stable order, so consecutive requests
        share a cacheable prompt prefix, followed by semantic facts and then episodic memories in
        chronological order.
        """
        if order == "score":
            return list(self.memories)
        if order == "chronological":
            return sorted(self.memories, key=lambda m: (m.timestamp, m.id))
        if order == "static_first":
            return sorted(
                self.memories,
                key=lambda m: (
                    _KIND_ORDER[m.kind],
                    0.0 if m.kind is MemoryKind.STATIC else m.timestamp,
                    m.id,
                ),
            )
        raise ValueError(f"unknown order: {order}")

    def render(self, order: Order = "static_first", separator: str = "\n") -> str:
        return separator.join(f"- {m.text}" for m in self.ordered(order))


@runtime_checkable
class Selector(Protocol):
    name: str

    def select(self, scored: Sequence[Scored], tokens: TokenFn, budget: int | None) -> list[Memory]:
        """Pick memories from ``scored`` (best first) whose total tokens fit within ``budget``."""
        ...


def _candidates(scored: Sequence[Scored], min_score: float) -> list[Scored]:
    return [s for s in scored if s.score > min_score]


class GreedySelector:
    """Take memories in score order, skipping any that would overflow the budget."""

    name = "greedy"

    def __init__(self, min_score: float = 0.0) -> None:
        self.min_score = min_score

    def select(self, scored: Sequence[Scored], tokens: TokenFn, budget: int | None) -> list[Memory]:
        chosen: list[Memory] = []
        used = 0
        for s in _candidates(scored, self.min_score):
            cost = tokens(s.memory)
            if budget is None or used + cost <= budget:
                chosen.append(s.memory)
                used += cost
        return chosen


class DensitySelector:
    """Take memories in order of score per token, favoring short, relevant memories."""

    name = "density"

    def __init__(self, min_score: float = 0.0) -> None:
        self.min_score = min_score

    def select(self, scored: Sequence[Scored], tokens: TokenFn, budget: int | None) -> list[Memory]:
        candidates = _candidates(scored, self.min_score)
        by_density = sorted(
            candidates, key=lambda s: (-s.score / max(1, tokens(s.memory)), s.memory.id)
        )
        return GreedySelector(min_score=self.min_score).select(by_density, tokens, budget)


class KnapsackSelector:
    """Maximize total score under the budget with 0/1 knapsack dynamic programming.

    Token weights are bucketed into at most ``max_units`` capacity units (rounding weights up,
    so the result never exceeds the budget), and only the ``max_candidates`` best-scored memories
    are considered. Both bounds keep the DP cheap enough to run per request.
    """

    name = "knapsack"

    def __init__(
        self, min_score: float = 0.0, max_candidates: int = 100, max_units: int = 500
    ) -> None:
        if max_candidates <= 0 or max_units <= 0:
            raise ValueError("max_candidates and max_units must be positive")
        self.min_score = min_score
        self.max_candidates = max_candidates
        self.max_units = max_units

    def select(self, scored: Sequence[Scored], tokens: TokenFn, budget: int | None) -> list[Memory]:
        candidates = _candidates(scored, self.min_score)
        if budget is None:
            return [s.memory for s in candidates]

        items = [s for s in candidates if tokens(s.memory) <= budget][: self.max_candidates]
        if not items:
            return []

        resolution = max(1, math.ceil(budget / self.max_units))
        capacity = budget // resolution
        weights = [math.ceil(tokens(s.memory) / resolution) for s in items]

        best = [0.0] * (capacity + 1)
        keep: list[bytearray] = []
        for weight, item in zip(weights, items, strict=True):
            row = bytearray(capacity + 1)
            for c in range(capacity, weight - 1, -1):
                candidate = best[c - weight] + item.score
                if candidate > best[c]:
                    best[c] = candidate
                    row[c] = 1
            keep.append(row)

        chosen: set[int] = set()
        c = capacity
        for i in range(len(items) - 1, -1, -1):
            if keep[i][c]:
                chosen.add(i)
                c -= weights[i]
        return [items[i].memory for i in sorted(chosen)]


def make_selection(memories: Sequence[Memory], tokens: TokenFn, budget: int | None) -> Selection:
    return Selection(tuple(memories), sum(tokens(m) for m in memories), budget)
