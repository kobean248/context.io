"""Retriever interface: score every memory in a store against a query."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar, runtime_checkable

from contextio.memory import Memory, MemoryStore

T = TypeVar("T")


@dataclass(frozen=True)
class Query:
    text: str
    timestamp: float | None = None
    """When the query is asked. Defaults to the newest memory's timestamp."""
    tags: frozenset[str] = field(default_factory=frozenset)

    def now(self, store: MemoryStore) -> float:
        return store.latest_timestamp if self.timestamp is None else self.timestamp


@dataclass(frozen=True)
class Scored:
    memory: Memory
    score: float


@runtime_checkable
class Retriever(Protocol):
    name: str

    def score(self, query: Query, store: MemoryStore) -> list[Scored]:
        """Score every memory in ``store``, best first."""
        ...


def rank(scored: Iterable[Scored]) -> list[Scored]:
    """Sort by score, breaking ties toward newer memories and then by id for determinism."""
    return sorted(scored, key=lambda s: (-s.score, -s.memory.timestamp, s.memory.id))


class StoreCache(Generic[T]):
    """Per-store cache for derived indexes, rebuilt when memories are added."""

    def __init__(self, build: Callable[[MemoryStore], T]) -> None:
        self._build = build
        self._entries: weakref.WeakKeyDictionary[MemoryStore, tuple[int, T]] = (
            weakref.WeakKeyDictionary()
        )

    def get(self, store: MemoryStore) -> T:
        entry = self._entries.get(store)
        if entry is None or entry[0] != len(store):
            entry = (len(store), self._build(store))
            self._entries[store] = entry
        return entry[1]
