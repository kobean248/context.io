from contextio.retrieval import BM25Retriever, Query
from contextio.selection import KnapsackSelector
from contextio.strategies import FullContext, RankedStrategy, Strategy, baseline_strategies


def test_full_context_ignores_budget(store):
    selection = FullContext().select(Query("anything"), store, budget=10)
    assert len(selection.memories) == len(store)
    assert selection.tokens == store.total_tokens
    assert selection.tokens > selection.budget
    timestamps = [m.timestamp for m in selection.memories]
    assert timestamps == sorted(timestamps)


def test_ranked_strategy_respects_budget(store):
    strategy = RankedStrategy("bm25", BM25Retriever())
    selection = strategy.select(Query("PostgreSQL database"), store, budget=20)
    assert selection.tokens <= 20
    assert selection.ids[0] == "db-new"
    assert "bm25" in repr(strategy)


def test_ranked_strategy_with_custom_selector(store):
    strategy = RankedStrategy("bm25-knap", BM25Retriever(), KnapsackSelector())
    selection = strategy.select(Query("caching for ledger"), store, budget=1000)
    assert "cache" in selection.ids


def test_baseline_strategies_have_unique_names_and_run(store):
    strategies = baseline_strategies()
    names = [s.name for s in strategies]
    assert len(names) == len(set(names))
    for strategy in strategies:
        assert isinstance(strategy, Strategy)
        selection = strategy.select(Query("which database do I use?"), store, budget=40)
        if strategy.name != "full":
            assert selection.tokens <= 40
