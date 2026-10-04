import pytest

from contextio.memory import Memory, MemoryKind
from contextio.retrieval import Scored
from contextio.selection import (
    DensitySelector,
    GreedySelector,
    KnapsackSelector,
    Selector,
    make_selection,
)

SIZES = {"big": 700, "mid": 400, "small-a": 300, "small-b": 300, "tiny": 50, "zero": 10}


def _tokens(m: Memory) -> int:
    return SIZES[m.id]


def _scored(*pairs: tuple[str, float]) -> list[Scored]:
    return [Scored(Memory(id=i, text=i), s) for i, s in pairs]


RANKED = _scored(
    ("big", 0.95), ("mid", 0.9), ("small-a", 0.6), ("small-b", 0.55), ("tiny", 0.2), ("zero", 0.0)
)


def _ids(memories):
    return [m.id for m in memories]


@pytest.mark.parametrize("selector", [GreedySelector(), DensitySelector(), KnapsackSelector()])
def test_selectors_respect_budget_and_protocol(selector):
    assert isinstance(selector, Selector)
    for budget in (0, 50, 100, 500, 1000, 1200, 5000):
        chosen = selector.select(RANKED, _tokens, budget)
        assert sum(_tokens(m) for m in chosen) <= budget
        assert "zero" not in _ids(chosen)


def test_greedy_skips_items_that_do_not_fit():
    assert _ids(GreedySelector().select(RANKED, _tokens, 1000)) == ["big", "small-a"]


def test_density_prefers_score_per_token():
    assert _ids(DensitySelector().select(RANKED, _tokens, 1000)) == [
        "tiny",
        "mid",
        "small-a",
    ]


def test_knapsack_maximizes_total_score():
    chosen = KnapsackSelector().select(RANKED, _tokens, 1000)
    assert set(_ids(chosen)) == {"mid", "small-a", "small-b"}
    assert sum(_tokens(m) for m in chosen) == 1000


def test_unbounded_budget_takes_everything_above_min_score():
    for selector in (GreedySelector(), DensitySelector(), KnapsackSelector()):
        assert set(_ids(selector.select(RANKED, _tokens, None))) == set(SIZES) - {"zero"}
    assert _ids(GreedySelector(min_score=0.5).select(RANKED, _tokens, None)) == [
        "big",
        "mid",
        "small-a",
        "small-b",
    ]


def test_knapsack_validation():
    with pytest.raises(ValueError):
        KnapsackSelector(max_units=0)


def test_selection_ordering_and_render():
    memories = [
        Memory(id="ep-late", text="late turn", timestamp=9),
        Memory(id="z-static", text="static z", kind=MemoryKind.STATIC, timestamp=5),
        Memory(id="fact", text="a fact", kind=MemoryKind.SEMANTIC, timestamp=3),
        Memory(id="a-static", text="static a", kind=MemoryKind.STATIC, timestamp=7),
        Memory(id="ep-early", text="early turn", timestamp=1),
    ]
    selection = make_selection(memories, lambda m: 2, budget=100)

    assert selection.tokens == 10
    assert selection.ids == [m.id for m in memories]
    assert [m.id for m in selection.ordered("static_first")] == [
        "a-static",
        "z-static",
        "fact",
        "ep-early",
        "ep-late",
    ]
    assert selection.ordered("chronological")[0].id == "ep-early"
    assert selection.render().splitlines()[0] == "- static a"
    with pytest.raises(ValueError):
        selection.ordered("random")
