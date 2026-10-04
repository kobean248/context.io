"""Aggregate run records into summaries and render them."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from contextio.eval.runner import RunRecord
from contextio.metrics import efficiency


@dataclass(frozen=True)
class Summary:
    strategy: str
    budget: int | None
    query_kind: str
    """Query kind, or "all"."""
    n: int
    answerable_rate: float
    accuracy: float | None
    recall: float
    precision: float
    f1: float
    utilization: float
    stale_rate: float
    """Fraction of queries whose context included superseded information."""
    mean_tokens: float
    cache_hit_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    mean_cost_usd: float
    answerable_per_1k_tokens: float

    @property
    def budget_label(self) -> str:
        return "none" if self.budget is None else str(self.budget)

    def to_dict(self) -> dict:
        return asdict(self)


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize_group(key: tuple[str, int | None, str], group: list[RunRecord]) -> Summary:
    strategy, budget, kind = key
    graded = [r.correct for r in group if r.correct is not None]
    total_input = sum(r.input_tokens for r in group)
    answerable_rate = _mean([float(r.answerable) for r in group])
    mean_tokens = _mean([r.context_tokens for r in group])
    return Summary(
        strategy=strategy,
        budget=budget,
        query_kind=kind,
        n=len(group),
        answerable_rate=answerable_rate,
        accuracy=_mean([float(c) for c in graded]) if graded else None,
        recall=_mean([r.recall for r in group]),
        precision=_mean([r.precision for r in group]),
        f1=_mean([r.f1 for r in group]),
        utilization=_mean([r.utilization for r in group]),
        stale_rate=_mean([float(r.stale_selected > 0) for r in group]),
        mean_tokens=mean_tokens,
        cache_hit_rate=sum(r.cached_tokens for r in group) / total_input if total_input else 0.0,
        p50_latency_ms=_percentile([r.latency_ms for r in group], 0.5),
        p95_latency_ms=_percentile([r.latency_ms for r in group], 0.95),
        mean_cost_usd=_mean([r.cost_usd for r in group]),
        answerable_per_1k_tokens=efficiency(answerable_rate, mean_tokens),
    )


def summarize(records: Iterable[RunRecord], by_kind: bool = False) -> list[Summary]:
    """One summary per (strategy, budget), plus per query kind when ``by_kind`` is set."""
    groups: dict[tuple[str, int | None, str], list[RunRecord]] = defaultdict(list)
    order: dict[str, int] = {}
    for record in records:
        order.setdefault(record.strategy, len(order))
        groups[(record.strategy, record.budget, "all")].append(record)
        if by_kind:
            groups[(record.strategy, record.budget, record.query_kind)].append(record)

    def sort_key(key: tuple[str, int | None, str]):
        strategy, budget, kind = key
        return (order[strategy], -1 if budget is None else budget, kind != "all", kind)

    return [_summarize_group(key, groups[key]) for key in sorted(groups, key=sort_key)]


def tokens_to_reach(
    summaries: Iterable[Summary], target: float, metric: str = "answerable_rate"
) -> dict[str, float | None]:
    """Fewest mean context tokens at which each strategy reaches ``target`` on ``metric``."""
    best: dict[str, float | None] = {}
    for s in summaries:
        if s.query_kind != "all":
            continue
        best.setdefault(s.strategy, None)
        value = getattr(s, metric)
        if value is not None and value >= target:
            current = best[s.strategy]
            best[s.strategy] = s.mean_tokens if current is None else min(current, s.mean_tokens)
    return best


DEFAULT_COLUMNS = (
    "strategy",
    "budget",
    "answerable_rate",
    "recall",
    "precision",
    "utilization",
    "stale_rate",
    "mean_tokens",
    "cache_hit_rate",
    "p50_latency_ms",
    "mean_cost_usd",
    "answerable_per_1k_tokens",
)

_PERCENT = {
    "answerable_rate",
    "accuracy",
    "recall",
    "precision",
    "f1",
    "utilization",
    "stale_rate",
    "cache_hit_rate",
}


def _format(column: str, value) -> str:
    if value is None:
        return "-" if column != "budget" else "none"
    if column in _PERCENT:
        return f"{value:.1%}"
    if column == "mean_cost_usd":
        return f"${value:.5f}"
    if column in {"mean_tokens"}:
        return f"{value:,.0f}"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def to_markdown(summaries: Iterable[Summary], columns: Sequence[str] = DEFAULT_COLUMNS) -> str:
    rows = [[_format(c, getattr(s, c)) for c in columns] for s in summaries]
    header = "| " + " | ".join(columns) + " |"
    align = "|" + "|".join("---" if c in {"strategy", "query_kind"} else "--:" for c in columns)
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, align + "|", *body])


def write_records_csv(records: Sequence[RunRecord], path: str | Path) -> None:
    names = [f.name for f in fields(RunRecord)]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())


def write_summaries_json(summaries: Sequence[Summary], path: str | Path) -> None:
    Path(path).write_text(json.dumps([s.to_dict() for s in summaries], indent=2) + "\n")
