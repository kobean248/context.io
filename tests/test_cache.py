import pytest

from contextio.cache import PrefixCache
from contextio.memory import Memory, MemoryKind
from contextio.selection import make_selection

PROFILE = [
    Memory(id="p-lang", text="langs", kind=MemoryKind.STATIC),
    Memory(id="p-name", text="name", kind=MemoryKind.STATIC),
]


def _tokens(m: Memory) -> int:
    return 10


def _selection(*episodic_ids: str):
    episodic = [Memory(id=i, text=i, timestamp=n) for n, i in enumerate(episodic_ids)]
    return make_selection([*episodic, *PROFILE], _tokens, budget=None)


def test_static_prefix_is_cached_across_requests():
    cache = PrefixCache()
    first = cache.observe("u1", _selection("a", "b"), _tokens)
    assert (first.cached_tokens, first.total_tokens) == (0, 40)

    second = cache.observe("u1", _selection("c"), _tokens)
    assert (second.cached_tokens, second.total_tokens) == (20, 30)
    assert second.hit_rate == pytest.approx(2 / 3)
    assert cache.hit_rate == pytest.approx(20 / 70)


def test_cache_is_per_key_and_resettable():
    cache = PrefixCache()
    cache.observe("u1", _selection("a"), _tokens)
    assert cache.observe("u2", _selection("a"), _tokens).cached_tokens == 0
    cache.reset()
    assert cache.hit_rate == 0.0
    assert cache.observe("u1", _selection("a"), _tokens).cached_tokens == 0


def test_min_prefix_tokens_threshold():
    cache = PrefixCache(min_prefix_tokens=25)
    cache.observe("u1", _selection("a"), _tokens)
    assert cache.observe("u1", _selection("b"), _tokens).cached_tokens == 0
    assert cache.observe("u1", _selection("b"), _tokens).cached_tokens == 30
