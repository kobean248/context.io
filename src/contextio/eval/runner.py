"""Run context strategies over a benchmark dataset across token budgets."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

from contextio.bench.dataset import BenchQuery, BenchUser, Dataset
from contextio.cache import PrefixCache
from contextio.consolidation import CompressedStrategy
from contextio.memory import MemoryStore
from contextio.metrics import evaluate_selection
from contextio.pricing import PricingModel
from contextio.retrieval import HybridRetriever
from contextio.selection import Selection
from contextio.strategies import (
    FullContext,
    RankedStrategy,
    Strategy,
    TwoStageStrategy,
    baseline_strategies,
)
from contextio.tokens import TokenCounter

DEFAULT_BUDGETS: tuple[int, ...] = (250, 500, 1000, 2000, 4000, 8000)


def standard_strategies() -> list[Strategy]:
    """Baselines plus compressed-memory variants."""
    return [
        *baseline_strategies(),
        CompressedStrategy(FullContext()),
        CompressedStrategy(RankedStrategy("hybrid", HybridRetriever(pin_static=True))),
    ]


@dataclass(frozen=True)
class RunRecord:
    strategy: str
    budget: int | None
    user_id: str
    query_id: str
    query_kind: str
    precision: float
    recall: float
    f1: float
    utilization: float
    answerable: bool
    stale_selected: int
    context_tokens: int
    input_tokens: int
    cached_tokens: int
    latency_ms: float
    cost_usd: float
    n_selected: int
    correct: bool | None = None
    """Whether a model's answer was graded correct; None when no model was run."""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkConfig:
    budgets: Sequence[int] = DEFAULT_BUDGETS
    pricing: PricingModel = field(default_factory=PricingModel)
    output_tokens: int = 300
    """Assumed answer length, used only for cost estimates."""
    simulate_cache: bool = True
    token_counter: TokenCounter | None = None


Answerer = Callable[[BenchUser, BenchQuery, Selection], bool]
"""Optional hook that answers a query from the selected context and grades it."""
Progress = Callable[[int, int], None]


def _timed_selections(
    strategy: Strategy, query: BenchQuery, store: MemoryStore, budgets: Sequence[int | None]
) -> list[tuple[Selection, float]]:
    q = query.to_query()
    if not isinstance(strategy, TwoStageStrategy):
        results = []
        for budget in budgets:
            start = time.perf_counter()
            selection = strategy.select(q, store, budget)
            results.append((selection, (time.perf_counter() - start) * 1000))
        return results

    start = time.perf_counter()
    scored = strategy.rank(q, store)
    rank_ms = (time.perf_counter() - start) * 1000
    results = []
    for budget in budgets:
        start = time.perf_counter()
        selection = strategy.choose(scored, store, budget)
        results.append((selection, rank_ms + (time.perf_counter() - start) * 1000))
    return results


def run_benchmark(
    dataset: Dataset,
    strategies: Sequence[Strategy] | None = None,
    config: BenchmarkConfig | None = None,
    answerer: Answerer | None = None,
    progress: Progress | None = None,
) -> list[RunRecord]:
    """Evaluate every strategy on every query at every budget.

    Strategies that ignore the budget (``uses_budget = False``) run once with ``budget=None``.
    Each (strategy, budget) pair gets its own prefix cache, so cache hits reflect consecutive
    queries from the same user under that configuration.
    """
    strategies = list(strategies) if strategies is not None else standard_strategies()
    config = config or BenchmarkConfig()
    names = [s.name for s in strategies]
    if len(set(names)) != len(names):
        raise ValueError(f"strategy names must be unique: {names}")

    caches: dict[tuple[str, int | None], PrefixCache] = {}
    records: list[RunRecord] = []
    for done, user in enumerate(dataset.users):
        store = user.store(config.token_counter)
        counter = store.token_counter
        for strategy in strategies:
            budgets: list[int | None] = (
                list(config.budgets) if getattr(strategy, "uses_budget", True) else [None]
            )
            for query in user.queries:
                query_tokens = counter(query.text)
                for budget, (selection, latency_ms) in zip(
                    budgets, _timed_selections(strategy, query, store, budgets), strict=True
                ):
                    metrics = evaluate_selection(
                        selection.memories, query.required_ids, store.tokens, query.stale_ids
                    )
                    cached = 0
                    if config.simulate_cache:
                        cache = caches.setdefault((strategy.name, budget), PrefixCache())
                        cached = cache.observe(user.id, selection, store.tokens).cached_tokens
                    input_tokens = selection.tokens + query_tokens
                    records.append(
                        RunRecord(
                            strategy=strategy.name,
                            budget=budget,
                            user_id=user.id,
                            query_id=query.id,
                            query_kind=query.kind,
                            precision=metrics.precision,
                            recall=metrics.recall,
                            f1=metrics.f1,
                            utilization=metrics.utilization,
                            answerable=metrics.answerable,
                            stale_selected=metrics.stale_selected,
                            context_tokens=selection.tokens,
                            input_tokens=input_tokens,
                            cached_tokens=cached,
                            latency_ms=latency_ms,
                            cost_usd=config.pricing.cost(
                                input_tokens, config.output_tokens, cached_input_tokens=cached
                            ),
                            n_selected=metrics.n_selected,
                            correct=answerer(user, query, selection) if answerer else None,
                        )
                    )
        if progress:
            progress(done + 1, len(dataset.users))
    return records
