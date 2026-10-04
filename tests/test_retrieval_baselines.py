import pytest

from contextio.memory import Memory, MemoryStore
from contextio.retrieval import BM25Retriever, Query, RecencyRetriever, Retriever, recency_decay


def _ids(scored):
    return [s.memory.id for s in scored]


def test_retrievers_satisfy_protocol():
    assert isinstance(BM25Retriever(), Retriever)
    assert isinstance(RecencyRetriever(), Retriever)


def test_recency_rank_mode_orders_newest_first(store):
    scored = RecencyRetriever().score(Query("anything"), store)
    assert _ids(scored)[:3] == ["weather", "db-new", "filler"]
    assert scored[0].score == 1.0
    assert scored[1].score == 0.5
    assert len(scored) == len(store)


def test_recency_half_life_mode_decays(store):
    scored = RecencyRetriever(half_life=10).score(Query("q", timestamp=60), store)
    by_id = {s.memory.id: s.score for s in scored}
    assert by_id["weather"] == pytest.approx(1.0)
    assert by_id["db-new"] == pytest.approx(0.5)
    assert by_id["db-old"] == pytest.approx(0.5**5)


def test_recency_decay_validation():
    assert recency_decay(-5, 10) == 1.0
    with pytest.raises(ValueError):
        recency_decay(1, 0)
    with pytest.raises(ValueError):
        RecencyRetriever(half_life=0)


def test_bm25_finds_keyword_matches(store):
    scored = BM25Retriever().score(
        Query("Which database should I use? PostgreSQL or MySQL?"), store
    )
    top_two = set(_ids(scored)[:2])
    assert top_two == {"db-new", "db-old"}
    assert scored[-1].score == 0.0


def test_bm25_prefers_decision_over_generic_filler(store):
    scored = BM25Retriever().score(Query("caching decision for the ledger project"), store)
    assert _ids(scored)[0] == "cache"


def test_bm25_index_refreshes_when_store_grows(store):
    retriever = BM25Retriever()
    assert retriever.score(Query("kubernetes"), store)[0].score == 0.0
    store.add(Memory(id="k8s", text="We deploy everything on Kubernetes.", timestamp=70))
    assert _ids(retriever.score(Query("kubernetes"), store))[0] == "k8s"


def test_bm25_handles_empty_store():
    assert BM25Retriever().score(Query("x"), MemoryStore()) == []
