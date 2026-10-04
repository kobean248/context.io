import pytest

from contextio.memory import Memory, MemoryKind, MemoryStore, heuristic_importance
from contextio.text import char_ngrams, jaccard, tokenize, words
from contextio.tokens import HeuristicTokenCounter, TokenCounter, default_counter


def test_memory_roundtrips_through_dict():
    m = Memory(
        id="m1",
        text="User prefers PostgreSQL.",
        kind=MemoryKind.SEMANTIC,
        timestamp=12.5,
        importance=0.9,
        tags=frozenset({"db"}),
        source_ids=("t1", "t2"),
    )
    assert Memory.from_dict(m.to_dict()) == m


def test_memory_coerces_collections_and_kind():
    m = Memory(id="m", text="x", kind="static", tags={"a"}, source_ids=["s"])
    assert m.kind is MemoryKind.STATIC
    assert m.tags == frozenset({"a"})
    assert m.source_ids == ("s",)
    assert m.covers == frozenset({"m", "s"})


def test_memory_rejects_out_of_range_importance():
    with pytest.raises(ValueError):
        Memory(id="m", text="x", importance=1.5)


def test_store_caches_tokens_and_tracks_usage():
    calls = []

    def counter(text: str) -> int:
        calls.append(text)
        return len(text.split())

    store = MemoryStore(
        [
            Memory(id="a", text="one two three", timestamp=1),
            Memory(id="b", text="four", timestamp=5),
        ],
        token_counter=counter,
    )
    a = store.get("a")
    assert store.tokens(a) == 3
    assert store.tokens(a) == 3
    assert calls.count("one two three") == 1
    assert store.total_tokens == 4
    assert store.latest_timestamp == 5
    assert len(store) == 2 and "a" in store

    store.record_use(["a", "a", "b"])
    assert store.usage("a") == 2
    assert store.max_usage == 2


def test_store_rejects_duplicate_ids():
    store = MemoryStore([Memory(id="a", text="x")])
    with pytest.raises(KeyError):
        store.add(Memory(id="a", text="y"))


def test_heuristic_token_counter_is_reasonable():
    counter = HeuristicTokenCounter()
    assert isinstance(counter, TokenCounter)
    assert counter("") == 0
    assert counter("hello world") == 2
    assert counter("internationalization") == 4
    assert counter("a, b.") == 4
    assert isinstance(default_counter(), HeuristicTokenCounter)


def test_heuristic_token_counter_validates_ratio():
    with pytest.raises(ValueError):
        HeuristicTokenCounter(chars_per_token=0)


def test_text_tokenization():
    assert words("I write C++ and C# code") == ["i", "write", "c++", "and", "c#", "code"]
    assert tokenize("The databases and queries") == ["database", "query"]
    assert char_ngrams("go") == ["<go", "go>"]
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert jaccard(set(), set()) == 1.0


def test_heuristic_importance_rewards_preferences_and_decisions():
    chit_chat = heuristic_importance("The weather was nice today.")
    preference = heuristic_importance("I prefer PostgreSQL for most things.")
    decision = heuristic_importance("We decided to use Redis for caching on our project.")
    settled = heuristic_importance("Okay, settled: the app will use Redis for caching.")
    assert chit_chat < preference
    assert chit_chat < decision
    assert chit_chat < settled
    assert 0.0 <= decision <= 1.0
