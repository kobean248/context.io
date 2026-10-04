"""Context strategies: end-to-end policies that turn (query, store, budget) into a Selection."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from contextio.memory import MemoryStore
from contextio.retrieval import (
    BM25Retriever,
    HybridRetriever,
    Query,
    RecencyRetriever,
    Retriever,
    Scored,
    VectorRetriever,
)
from contextio.selection import (
    GreedySelector,
    KnapsackSelector,
    Selection,
    Selector,
    make_selection,
)


@runtime_checkable
class Strategy(Protocol):
    name: str

    def select(self, query: Query, store: MemoryStore, budget: int | None) -> Selection: ...


@runtime_checkable
class TwoStageStrategy(Strategy, Protocol):
    """A strategy whose budget-independent ranking can be reused across budgets."""

    def rank(self, query: Query, store: MemoryStore) -> list[Scored]: ...

    def choose(self, scored: list[Scored], store: MemoryStore, budget: int | None) -> Selection: ...


class FullContext:
    """Send the entire history, ignoring the budget. The accuracy upper bound and cost ceiling."""

    name = "full"
    uses_budget = False

    def rank(self, query: Query, store: MemoryStore) -> list[Scored]:
        return [Scored(m, 1.0) for m in sorted(store, key=lambda m: (m.timestamp, m.id))]

    def choose(self, scored: list[Scored], store: MemoryStore, budget: int | None) -> Selection:
        return make_selection([s.memory for s in scored], store.tokens, budget)

    def select(self, query: Query, store: MemoryStore, budget: int | None) -> Selection:
        return self.choose(self.rank(query, store), store, budget)


class RankedStrategy:
    """Score memories with a retriever, then fit them into the budget with a selector."""

    def __init__(self, name: str, retriever: Retriever, selector: Selector | None = None) -> None:
        self.name = name
        self.retriever = retriever
        self.selector = selector or GreedySelector()

    def rank(self, query: Query, store: MemoryStore) -> list[Scored]:
        return self.retriever.score(query, store)

    def choose(self, scored: list[Scored], store: MemoryStore, budget: int | None) -> Selection:
        chosen = self.selector.select(scored, store.tokens, budget)
        return make_selection(chosen, store.tokens, budget)

    def select(self, query: Query, store: MemoryStore, budget: int | None) -> Selection:
        return self.choose(self.rank(query, store), store, budget)

    def __repr__(self) -> str:
        return (
            f"RankedStrategy(name={self.name!r}, retriever={self.retriever.name}, "
            f"selector={self.selector.name})"
        )


def baseline_strategies() -> list[Strategy]:
    """The standard comparison set: full context, recency, keyword, vector, and hybrid."""
    return [
        FullContext(),
        RankedStrategy("recency", RecencyRetriever()),
        RankedStrategy("bm25", BM25Retriever()),
        RankedStrategy("vector", VectorRetriever()),
        RankedStrategy("hybrid", HybridRetriever(pin_static=True)),
        RankedStrategy("hybrid-knapsack", HybridRetriever(pin_static=True), KnapsackSelector()),
    ]
