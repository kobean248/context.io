import json

import pytest

from contextio.consolidation import (
    CompressedStrategy,
    Consolidator,
    LLMConsolidator,
    RuleBasedConsolidator,
    deduplicate,
    split_sentences,
    to_third_person,
)
from contextio.memory import Memory, MemoryKind, MemoryStore
from contextio.metrics import evaluate_selection
from contextio.retrieval import BM25Retriever, Query
from contextio.strategies import RankedStrategy


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("I prefer PostgreSQL.", "User prefers PostgreSQL."),
        ("I've switched to Go.", "User has switched to Go."),
        ("I'm starting my new project.", "User is starting their new project."),
        (
            "we decided to use Redis for our cache.",
            "The team decided to use Redis for their cache.",
        ),
        ("I mostly use MySQL.", "User mostly uses MySQL."),
    ],
)
def test_to_third_person(sentence, expected):
    assert to_third_person(sentence) == expected


def test_split_sentences():
    assert split_sentences("Hi there! How are you? Fine.") == ["Hi there!", "How are you?", "Fine."]


def test_deduplicate_keeps_newest_and_merges_sources():
    memories = [
        Memory(id="a", text="User prefers dark mode", timestamp=1, importance=0.9),
        Memory(id="b", text="User prefers dark mode!", timestamp=5, importance=0.4),
        Memory(id="c", text="User deploys on Fridays", timestamp=3),
    ]
    result = deduplicate(memories)
    assert [m.id for m in result] == ["c", "b"]
    merged = result[1]
    assert merged.covers == {"a", "b"}
    assert merged.importance == 0.9


def test_rule_based_consolidator_extracts_facts_and_drops_chatter():
    memories = [
        Memory(id="profile", text="Name: Priya.", kind=MemoryKind.STATIC),
        Memory(
            id="t1",
            text="Hey, hope your week is going well. I've switched to PostgreSQL for everything "
            "now. Anyway, the weather is nice.",
            timestamp=10,
        ),
        Memory(id="t2", text="The weather is nice and sunny.", timestamp=11),
    ]
    consolidator = RuleBasedConsolidator()
    assert isinstance(consolidator, Consolidator)
    result = consolidator.consolidate(memories)

    ids = [m.id for m in result]
    assert ids == ["profile", "t1#f0"]
    fact = result[1]
    assert fact.kind is MemoryKind.SEMANTIC
    assert fact.text == "User has switched to PostgreSQL for everything now."
    assert fact.source_ids == ("t1",)

    kept = RuleBasedConsolidator(keep_unmatched=True).consolidate(memories)
    assert "t2" in [m.id for m in kept]


def test_llm_consolidator_parses_model_output():
    memories = [
        Memory(id="t1", text="I use MySQL.", timestamp=1),
        Memory(id="t2", text="Switched to Postgres.", timestamp=9),
        Memory(id="p", text="Name: Priya", kind=MemoryKind.STATIC),
    ]
    prompts = []

    def fake_complete(prompt: str) -> str:
        prompts.append(prompt)
        return "Sure!\n" + json.dumps(
            [
                {"fact": "User prefers PostgreSQL.", "source_ids": ["t2"], "importance": 0.9},
                {"fact": "Hallucinated", "source_ids": ["nope"]},
                {"fact": "", "source_ids": ["t1"]},
            ]
        )

    result = LLMConsolidator(fake_complete).consolidate(memories)
    assert len(prompts) == 1
    assert "[t1]" in prompts[0] and "[t2]" in prompts[0]
    assert [m.text for m in result] == ["Name: Priya", "User prefers PostgreSQL."]
    assert result[1].timestamp == 9
    assert result[1].source_ids == ("t2",)


def test_llm_consolidator_tolerates_bad_output_and_validates():
    memories = [Memory(id="t1", text="x")]
    assert LLMConsolidator(lambda p: "not json").consolidate(memories) == []
    assert LLMConsolidator(lambda p: "[1, 2]").consolidate(memories) == []
    with pytest.raises(ValueError):
        LLMConsolidator(lambda p: "", batch_size=0)


def test_compressed_strategy_reduces_tokens_and_keeps_recall(store):
    inner = RankedStrategy("bm25", BM25Retriever())
    compressed = CompressedStrategy(inner)
    assert compressed.name == "compressed-bm25"

    assert compressed.consolidated(store).total_tokens < store.total_tokens

    small = compressed.select(Query("Which database should I use?"), store, budget=None)
    assert evaluate_selection(small.memories, {"db-new"}, store.tokens).answerable
    assert compressed.consolidated(store) is compressed.consolidated(store)


def test_compressed_strategy_rebuilds_when_store_grows():
    store = MemoryStore([Memory(id="a", text="I prefer tea.", timestamp=1)])
    strategy = CompressedStrategy(RankedStrategy("bm25", BM25Retriever()))
    assert len(strategy.consolidated(store)) == 1
    store.add(Memory(id="b", text="We decided to ship on Monday.", timestamp=2))
    assert len(strategy.consolidated(store)) == 2
