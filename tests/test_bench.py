import pytest

from contextio.bench import BenchQuery, Dataset, generate_dataset
from contextio.bench.synthetic import generate_user
from contextio.memory import MemoryKind


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return generate_dataset(n_users=8, seed=7)


def test_generation_is_deterministic():
    a = generate_dataset(n_users=3, seed=1)
    b = generate_dataset(n_users=3, seed=1)
    c = generate_dataset(n_users=3, seed=2)
    assert [u.to_dict() for u in a] == [u.to_dict() for u in b]
    assert [u.to_dict() for u in a] != [u.to_dict() for u in c]


def test_ground_truth_references_existing_memories(dataset):
    for user in dataset:
        ids = {m.id for m in user.memories}
        assert len(ids) == len(user.memories)
        for query in user.queries:
            assert query.required_ids
            assert query.required_ids <= ids
            assert query.stale_ids <= ids
            assert not query.required_ids & query.stale_ids


def test_required_memories_precede_the_query(dataset):
    for user in dataset:
        by_id = {m.id: m for m in user.memories}
        for query in user.queries:
            assert all(by_id[r].timestamp < query.timestamp for r in query.required_ids)


def test_stale_memories_are_older_than_their_replacements(dataset):
    for user in dataset:
        by_id = {m.id: m for m in user.memories}
        for query in user.queries:
            if query.stale_ids:
                newest_required = max(by_id[r].timestamp for r in query.required_ids)
                assert all(by_id[s].timestamp < newest_required for s in query.stale_ids)


def test_answer_keywords_appear_in_required_memories(dataset):
    for user in dataset:
        by_id = {m.id: m for m in user.memories}
        for query in user.queries:
            evidence = " ".join(by_id[r].text for r in query.required_ids)
            for keyword in query.answer_keywords:
                assert keyword in evidence


def test_query_mix_and_history_shape(dataset):
    kinds = {q.kind for u in dataset for q in u.queries}
    assert kinds == {"background", "style", "preference", "continuation", "decision"}
    user = dataset.users[0]
    static = [m for m in user.memories if m.kind is MemoryKind.STATIC]
    assert len(static) == 7
    timestamps = [m.timestamp for m in user.memories if m.kind is MemoryKind.EPISODIC]
    assert timestamps == sorted(timestamps)
    assert user.store().total_tokens > 5000


def test_stale_cases_are_present(dataset):
    assert any(q.stale_ids for u in dataset for q in u.queries if q.kind == "preference")
    assert any(q.stale_ids for u in dataset for q in u.queries if q.kind == "continuation")


def test_dataset_jsonl_roundtrip(tmp_path, dataset):
    path = tmp_path / "bench.jsonl"
    dataset.save_jsonl(path)
    loaded = Dataset.load_jsonl(path)
    assert loaded.name == "bench"
    assert [u.to_dict() for u in loaded] == [u.to_dict() for u in dataset]
    assert loaded.n_queries == dataset.n_queries


def test_stats(dataset):
    stats = dataset.stats()
    assert stats["users"] == 8
    assert stats["queries"] == dataset.n_queries
    assert stats["avg_tokens_per_user"] > 0
    assert sum(stats["query_kinds"].values()) == dataset.n_queries


def test_bench_query_to_query():
    q = BenchQuery(id="q", text="hi", kind="style", required_ids=frozenset({"a"}), timestamp=3.0)
    assert q.to_query().text == "hi"
    assert q.to_query().timestamp == 3.0
    assert BenchQuery.from_dict(q.to_dict()) == q


def test_generate_user_and_validation():
    assert generate_user("solo").id == "solo"
    with pytest.raises(ValueError):
        generate_dataset(0)
