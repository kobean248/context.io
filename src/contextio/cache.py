"""Prompt-prefix cache simulation.

LLM providers bill repeated prompt prefixes at a discount. When context is rendered with stable
memories first (see ``Selection.ordered("static_first")``), consecutive requests from the same
user share a prefix. ``PrefixCache`` measures how many input tokens that saves.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from contextio.memory import Memory
from contextio.selection import Order, Selection


@dataclass(frozen=True)
class CacheResult:
    cached_tokens: int
    total_tokens: int

    @property
    def hit_rate(self) -> float:
        return self.cached_tokens / self.total_tokens if self.total_tokens else 0.0


class PrefixCache:
    """Tracks the last rendered context per key (usually a user id).

    On each request, the leading memories that match the previous request's leading memories
    count as cached. Prefixes shorter than ``min_prefix_tokens`` are not cached, mirroring
    providers that only cache prompts above a minimum length.
    """

    def __init__(self, min_prefix_tokens: int = 0, order: Order = "static_first") -> None:
        self.min_prefix_tokens = min_prefix_tokens
        self.order = order
        self._last: dict[str, list[str]] = {}
        self.cached_tokens = 0
        self.total_tokens = 0

    def observe(
        self, key: str, selection: Selection, tokens: Callable[[Memory], int]
    ) -> CacheResult:
        ordered = selection.ordered(self.order)
        previous = self._last.get(key, [])

        cached = 0
        for i, memory in enumerate(ordered):
            if i >= len(previous) or previous[i] != memory.id:
                break
            cached += tokens(memory)
        if cached < self.min_prefix_tokens:
            cached = 0

        total = sum(tokens(m) for m in ordered)
        self._last[key] = [m.id for m in ordered]
        self.cached_tokens += cached
        self.total_tokens += total
        return CacheResult(cached, total)

    @property
    def hit_rate(self) -> float:
        return self.cached_tokens / self.total_tokens if self.total_tokens else 0.0

    def reset(self) -> None:
        self._last.clear()
        self.cached_tokens = 0
        self.total_tokens = 0
