import pytest

from contextio.memory import Memory, MemoryKind, MemoryStore
from contextio.retrieval import (
    BM25Retriever,
    HybridRetriever,
    HybridWeights,
    Query,
    Retriever,
    Scored,
    normalize_scores,
)


class FixedRetriever:
    name = "fixed"

    def __init__(self, scores):
        self.scores = scores

    def score(self, query, store):
        return [Scored(m, self.scores.get(m.id, 0.0)) for m in store]


def _by_id(scored):
    return {s.memory.id: s.score for s in scored}


def test_normalize_scores():
    memories = [Memory(id=i, text=i) for i in "abc"]
    scored = [Scored(memories[0], 4.0), Scored(memories[1], 2.0), Scored(memories[2], -1.0)]
    assert normalize_scores(scored) == {"a": 1.0, "b": 0.5, "c": 0.0}
    assert normalize_scores([Scored(memories[0], 0.0)]) == {"a": 0.0}


def test_hybrid_combines_signals_with_weights():
    store = MemoryStore(
        [
            Memory(id="old-important", text="x", timestamp=0, importance=1.0),
            Memory(id="new-trivial", text="y", timestamp=30, importance=0.0),
        ]
    )
    retriever = HybridRetriever(
        similarity=FixedRetriever({"old-important": 1.0, "new-trivial": 1.0}),
        weights=HybridWeights(recency=1.0, similarity=0.0, importance=0.0, frequency=0.0),
        half_life=30.0,
    )
    assert isinstance(retriever, Retriever)
    scores = _by_id(retriever.score(Query("q", timestamp=30), store))
    assert scores["new-trivial"] == pytest.approx(1.0)
    assert scores["old-important"] == pytest.approx(0.5)

    retriever.weights = HybridWeights(recency=0.0, similarity=0.0, importance=1.0, frequency=0.0)
    scores = _by_id(retriever.score(Query("q", timestamp=30), store))
    assert scores == {"old-important": 1.0, "new-trivial": 0.0}


def test_frequency_signal_uses_store_usage():
    store = MemoryStore([Memory(id="a", text="a"), Memory(id="b", text="b")])
    store.record_use(["a", "a", "a", "b"])
    retriever = HybridRetriever(
        similarity=FixedRetriever({"a": 1.0, "b": 1.0}),
        weights=HybridWeights(recency=0.0, similarity=0.0, importance=0.0, frequency=1.0),
    )
    scores = _by_id(retriever.score(Query("q"), store))
    assert scores["a"] == pytest.approx(1.0)
    assert 0.0 < scores["b"] < 1.0


def test_similarity_gate_and_static_pinning():
    store = MemoryStore(
        [
            Memory(id="on-topic", text="a"),
            Memory(id="off-topic", text="b", importance=1.0),
            Memory(id="profile", text="c", kind=MemoryKind.STATIC, importance=1.0),
        ]
    )
    similarity = FixedRetriever({"on-topic": 1.0, "off-topic": 0.05, "profile": 0.0})

    gated = _by_id(HybridRetriever(similarity, min_similarity=0.1).score(Query("q"), store))
    assert gated["on-topic"] > 0
    assert gated["off-topic"] == 0.0
    assert gated["profile"] == 0.0

    pinned = _by_id(
        HybridRetriever(similarity, min_similarity=0.1, pin_static=True).score(Query("q"), store)
    )
    assert pinned["profile"] > 0
    assert pinned["off-topic"] == 0.0


def test_static_memories_do_not_decay():
    store = MemoryStore(
        [
            Memory(id="profile", text="p", kind=MemoryKind.STATIC, timestamp=0),
            Memory(id="turn", text="t", timestamp=0),
        ]
    )
    retriever = HybridRetriever(
        FixedRetriever({"profile": 1.0, "turn": 1.0}),
        weights=HybridWeights(recency=1.0, similarity=0.0, importance=0.0, frequency=0.0),
        half_life=1.0,
    )
    scores = _by_id(retriever.score(Query("q", timestamp=10), store))
    assert scores["profile"] == 1.0
    assert scores["turn"] < 0.01


def test_default_hybrid_on_realistic_store(store):
    scored = HybridRetriever().score(
        Query("what caching did we pick for the ledger project?"), store
    )
    assert scored[0].memory.id == "cache"


def test_hybrid_validation():
    with pytest.raises(ValueError):
        HybridRetriever(similarity=[])
    with pytest.raises(ValueError):
        HybridWeights(recency=-1)
    assert HybridRetriever(similarity=BM25Retriever()).retrievers[0].name == "bm25"
