"""Context quality and efficiency metrics.

All metrics compare the memories a strategy *selected* against the set of original memory ids
that are *required* to answer a query. A selected memory counts as relevant when it covers at
least one required id, either directly or through ``Memory.source_ids``; this lets consolidated
or compressed memories be scored against the same ground truth as raw ones.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass

from contextio.memory import Memory


@dataclass(frozen=True)
class ContextMetrics:
    precision: float
    """Fraction of selected memories that cover at least one required memory."""
    recall: float
    """Fraction of required memories covered by the selection."""
    f1: float
    utilization: float
    """Fraction of selected tokens that belong to relevant memories."""
    selected_tokens: int
    relevant_tokens: int
    n_selected: int
    n_relevant_selected: int
    n_required: int
    stale_selected: int
    """Number of selected memories that carry outdated, superseded information."""

    @property
    def answerable(self) -> bool:
        """Whether every required memory made it into the context."""
        return self.recall >= 1.0

    def to_dict(self) -> dict:
        return {**asdict(self), "answerable": self.answerable}


def evaluate_selection(
    selected: Sequence[Memory],
    required: Iterable[str],
    tokens: Callable[[Memory], int],
    stale: Iterable[str] = (),
) -> ContextMetrics:
    required_ids = frozenset(required)
    stale_ids = frozenset(stale)

    covered: set[str] = set()
    n_relevant = 0
    relevant_tokens = 0
    selected_tokens = 0
    stale_selected = 0
    for memory in selected:
        n_tokens = tokens(memory)
        selected_tokens += n_tokens
        hits = memory.covers & required_ids
        if hits:
            covered |= hits
            n_relevant += 1
            relevant_tokens += n_tokens
        if memory.covers & stale_ids and not hits:
            stale_selected += 1

    nothing_needed = not required_ids
    precision = n_relevant / len(selected) if selected else float(nothing_needed)
    recall = len(covered) / len(required_ids) if required_ids else 1.0
    utilization = relevant_tokens / selected_tokens if selected_tokens else float(nothing_needed)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return ContextMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        utilization=utilization,
        selected_tokens=selected_tokens,
        relevant_tokens=relevant_tokens,
        n_selected=len(selected),
        n_relevant_selected=n_relevant,
        n_required=len(required_ids),
        stale_selected=stale_selected,
    )


def efficiency(performance: float, tokens: float, per: float = 1000.0) -> float:
    """Task performance per ``per`` context tokens."""
    if tokens <= 0:
        return 0.0 if performance <= 0 else float("inf")
    return performance / (tokens / per)


def marginal_efficiency(
    performance_a: float,
    tokens_a: float,
    performance_b: float,
    tokens_b: float,
    per: float = 1000.0,
) -> float:
    """Change in performance per ``per`` extra tokens when moving from system A to system B."""
    delta_tokens = tokens_b - tokens_a
    if delta_tokens == 0:
        return 0.0
    return (performance_b - performance_a) / (delta_tokens / per)
