import csv
import json

import pytest

from contextio.bench import generate_dataset
from contextio.consolidation import CompressedStrategy
from contextio.eval import (
    BenchmarkConfig,
    run_benchmark,
    standard_strategies,
    summarize,
    to_markdown,
    tokens_to_reach,
    write_records_csv,
    write_summaries_json,
)
from contextio.memory import MemoryStore
from contextio.pricing import PricingModel
from contextio.retrieval import BM25Retriever, Query
from contextio.selection import make_selection
from contextio.strategies import FullContext, RankedStrategy, TwoStageStrategy


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(n_users=3, seed=11)


@pytest.fixture(scope="module")
def records(dataset):
    strategies = [
        FullContext(),
        RankedStrategy("bm25", BM25Retriever()),
        CompressedStrategy(FullContext()),
    ]
    config = BenchmarkConfig(budgets=(200, 2000))
    return run_benchmark(dataset, strategies, config)


def test_record_counts_respect_budget_awareness(dataset, records):
    n = dataset.n_queries
    by_strategy = {}
    for r in records:
        by_strategy.setdefault(r.strategy, set()).add(r.budget)
    assert by_strategy == {"full": {None}, "bm25": {200, 2000}, "compressed-full": {None}}
    assert len(records) == n * (1 + 2 + 1)


def test_records_are_consistent(records):
    for r in records:
        assert 0.0 <= r.recall <= 1.0
        assert r.answerable == (r.recall >= 1.0)
        assert r.input_tokens > r.context_tokens
        assert 0 <= r.cached_tokens <= r.context_tokens
        assert r.latency_ms >= 0
        assert r.cost_usd > 0
        assert r.correct is None
        if r.budget is not None:
            assert r.context_tokens <= r.budget


def test_full_context_is_always_answerable(records):
    assert all(r.answerable for r in records if r.strategy == "full")


def test_summaries_and_frontier(records):
    summaries = summarize(records)
    keys = [(s.strategy, s.budget) for s in summaries]
    assert keys == [("full", None), ("bm25", 200), ("bm25", 2000), ("compressed-full", None)]

    full = summaries[0]
    assert full.answerable_rate == 1.0
    assert full.accuracy is None
    assert full.cache_hit_rate > 0.5
    assert full.p50_latency_ms <= full.p95_latency_ms

    frontier = tokens_to_reach(summaries, target=1.0)
    assert frontier["full"] == pytest.approx(full.mean_tokens)
    assert set(frontier) == {"full", "bm25", "compressed-full"}


def test_summaries_by_kind(records, dataset):
    summaries = summarize(records, by_kind=True)
    kinds = {s.query_kind for s in summaries if s.strategy == "full"}
    assert kinds == {"all", "background", "style", "preference", "continuation", "decision"}
    full_kinds = [s for s in summaries if s.strategy == "full" and s.query_kind != "all"]
    assert sum(s.n for s in full_kinds) == dataset.n_queries


def test_markdown_and_file_outputs(records, tmp_path):
    summaries = summarize(records)
    table = to_markdown(summaries)
    lines = table.splitlines()
    assert lines[0].startswith("| strategy | budget |")
    assert len(lines) == 2 + len(summaries)
    assert "| full | none |" in table

    csv_path = tmp_path / "records.csv"
    write_records_csv(records, csv_path)
    with csv_path.open() as f:
        assert len(list(csv.DictReader(f))) == len(records)

    json_path = tmp_path / "summary.json"
    write_summaries_json(summaries, json_path)
    assert len(json.loads(json_path.read_text())) == len(summaries)


def test_answerer_hook_and_cost_config(dataset):
    calls = []

    def answerer(user, query, selection):
        calls.append(query.id)
        return True

    config = BenchmarkConfig(
        budgets=(500,),
        pricing=PricingModel(input_per_mtok=1.0, output_per_mtok=0.0, cached_input_per_mtok=None),
        output_tokens=0,
        simulate_cache=False,
    )
    records = run_benchmark(dataset, [FullContext()], config, answerer=answerer)
    assert len(calls) == dataset.n_queries
    assert all(r.correct for r in records)
    assert all(r.cached_tokens == 0 for r in records)
    assert all(r.cost_usd == pytest.approx(r.input_tokens / 1e6) for r in records)
    assert summarize(records)[0].accuracy == 1.0


def test_single_stage_strategies_are_supported(dataset):
    class OnlySelect:
        name = "first-memory"

        def select(self, query, store, budget):
            first = next(iter(store))
            return make_selection([first], store.tokens, budget)

    assert not isinstance(OnlySelect(), TwoStageStrategy)
    records = run_benchmark(dataset, [OnlySelect()], BenchmarkConfig(budgets=(100, 200)))
    assert len(records) == dataset.n_queries * 2


def test_progress_and_duplicate_names(dataset):
    seen = []
    run_benchmark(
        dataset,
        [FullContext()],
        BenchmarkConfig(budgets=(100,)),
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen[-1] == (3, 3)
    with pytest.raises(ValueError):
        run_benchmark(dataset, [FullContext(), FullContext()])


def test_standard_strategies_are_two_stage_with_unique_names():
    strategies = standard_strategies()
    assert len({s.name for s in strategies}) == len(strategies)
    assert all(isinstance(s, TwoStageStrategy) for s in strategies)


def test_two_stage_matches_select(store: MemoryStore):
    for strategy in standard_strategies():
        query = Query("what database should I use?")
        direct = strategy.select(query, store, 50)
        staged = strategy.choose(strategy.rank(query, store), store, 50)
        assert direct.ids == staged.ids
