import pytest

from contextio.memory import Memory, MemoryKind
from contextio.metrics import efficiency, evaluate_selection, marginal_efficiency
from contextio.pricing import PricingModel


def _tokens(m: Memory) -> int:
    return len(m.text.split())


def test_precision_recall_and_utilization():
    selected = [
        Memory(id="a", text="one two three four"),
        Memory(id="b", text="five six"),
        Memory(id="noise", text="x " * 14),
    ]
    m = evaluate_selection(selected, required={"a", "b", "c"}, tokens=_tokens)

    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 3)
    assert m.f1 == pytest.approx(2 / 3)
    assert m.selected_tokens == 20
    assert m.relevant_tokens == 6
    assert m.utilization == pytest.approx(0.3)
    assert not m.answerable


def test_consolidated_memory_covers_its_sources():
    fact = Memory(
        id="fact",
        text="User prefers Go.",
        kind=MemoryKind.SEMANTIC,
        source_ids=("turn-3", "turn-9"),
    )
    m = evaluate_selection([fact], required={"turn-3", "turn-9"}, tokens=_tokens)
    assert m.recall == 1.0
    assert m.precision == 1.0
    assert m.answerable


def test_stale_memories_are_counted():
    selected = [Memory(id="old", text="uses mysql"), Memory(id="new", text="uses postgres")]
    m = evaluate_selection(selected, required={"new"}, tokens=_tokens, stale={"old"})
    assert m.stale_selected == 1
    assert m.answerable


def test_empty_selection_edge_cases():
    nothing_needed = evaluate_selection([], required=set(), tokens=_tokens)
    assert (nothing_needed.precision, nothing_needed.recall, nothing_needed.utilization) == (
        1.0,
        1.0,
        1.0,
    )

    missed = evaluate_selection([], required={"a"}, tokens=_tokens)
    assert (missed.precision, missed.recall, missed.utilization, missed.f1) == (0.0, 0.0, 0.0, 0.0)


def test_efficiency_metrics():
    assert efficiency(0.9, 4000) == pytest.approx(0.225)
    assert efficiency(0.89, 1000) == pytest.approx(0.89)
    assert efficiency(0.0, 0) == 0.0
    assert marginal_efficiency(0.89, 1000, 0.90, 4000) == pytest.approx(0.01 / 3)
    assert marginal_efficiency(0.5, 100, 0.6, 100) == 0.0


def test_pricing_cost_with_cache_and_output():
    pricing = PricingModel(input_per_mtok=3.0, output_per_mtok=15.0, cached_input_per_mtok=0.3)
    assert pricing.cost(1_000_000) == pytest.approx(3.0)
    assert pricing.cost(1_000_000, output_tokens=1_000_000) == pytest.approx(18.0)
    assert pricing.cost(1_000_000, cached_input_tokens=500_000) == pytest.approx(1.65)


def test_pricing_without_cache_discount_and_validation():
    pricing = PricingModel(cached_input_per_mtok=None)
    assert pricing.cost(1000, cached_input_tokens=1000) == pricing.cost(1000)
    with pytest.raises(ValueError):
        pricing.cost(10, cached_input_tokens=11)
    with pytest.raises(ValueError):
        PricingModel(input_per_mtok=-1)
