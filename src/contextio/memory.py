"""Memory items and the per-user memory store."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum

from contextio.tokens import TokenCounter, default_counter


class MemoryKind(str, Enum):
    """How a piece of context behaves over time.

    STATIC memories rarely change (name, languages, long-term preferences) and are good
    candidates for aggressive caching. EPISODIC memories are tied to a moment (a conversation
    turn, a status update) and need dynamic retrieval. SEMANTIC memories are consolidated facts
    distilled from episodic ones.
    """

    STATIC = "static"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


@dataclass(frozen=True)
class Memory:
    id: str
    text: str
    kind: MemoryKind = MemoryKind.EPISODIC
    timestamp: float = 0.0
    importance: float = 0.5
    tags: frozenset[str] = field(default_factory=frozenset)
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError(f"importance must be in [0, 1], got {self.importance}")
        if not isinstance(self.kind, MemoryKind):
            object.__setattr__(self, "kind", MemoryKind(self.kind))
        if not isinstance(self.tags, frozenset):
            object.__setattr__(self, "tags", frozenset(self.tags))
        if not isinstance(self.source_ids, tuple):
            object.__setattr__(self, "source_ids", tuple(self.source_ids))

    @property
    def covers(self) -> frozenset[str]:
        """Ids of the original memories whose information this memory carries."""
        return frozenset((self.id, *self.source_ids))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "kind": self.kind.value,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "tags": sorted(self.tags),
            "source_ids": list(self.source_ids),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Memory:
        return cls(
            id=data["id"],
            text=data["text"],
            kind=MemoryKind(data.get("kind", MemoryKind.EPISODIC.value)),
            timestamp=float(data.get("timestamp", 0.0)),
            importance=float(data.get("importance", 0.5)),
            tags=frozenset(data.get("tags", ())),
            source_ids=tuple(data.get("source_ids", ())),
        )


class MemoryStore:
    """An ordered collection of a single user's memories.

    The store caches token counts and tracks how often each memory has been used, which feeds
    frequency-based scoring.
    """

    def __init__(
        self, memories: Iterable[Memory] = (), token_counter: TokenCounter | None = None
    ) -> None:
        self.token_counter = token_counter or default_counter()
        self._memories: dict[str, Memory] = {}
        self._tokens: dict[str, int] = {}
        self._usage: Counter[str] = Counter()
        for memory in memories:
            self.add(memory)

    def add(self, memory: Memory) -> None:
        if memory.id in self._memories:
            raise KeyError(f"duplicate memory id: {memory.id}")
        self._memories[memory.id] = memory

    def get(self, memory_id: str) -> Memory:
        return self._memories[memory_id]

    def __contains__(self, memory_id: object) -> bool:
        return memory_id in self._memories

    def __iter__(self) -> Iterator[Memory]:
        return iter(self._memories.values())

    def __len__(self) -> int:
        return len(self._memories)

    def tokens(self, memory: Memory) -> int:
        count = self._tokens.get(memory.id)
        if count is None:
            count = self.token_counter(memory.text)
            self._tokens[memory.id] = count
        return count

    @property
    def total_tokens(self) -> int:
        return sum(self.tokens(m) for m in self)

    @property
    def latest_timestamp(self) -> float:
        return max((m.timestamp for m in self), default=0.0)

    def record_use(self, memory_ids: Iterable[str]) -> None:
        self._usage.update(memory_ids)

    def usage(self, memory_id: str) -> int:
        return self._usage[memory_id]

    @property
    def max_usage(self) -> int:
        return max(self._usage.values(), default=0)


_IMPORTANCE_CUES: tuple[tuple[re.Pattern[str], float], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), weight)
    for pattern, weight in (
        (r"\b(i|we)\s+(prefer|always|never|usually|mostly)\b", 0.35),
        (r"\b(decided|settled on|going with|chose|picked|agreed)\b", 0.35),
        (r"\b(switched|moved over|migrated|changed my mind|no longer)\b", 0.3),
        (r"\b(my|our)\s+(project|team|stack|company|role|job)\b", 0.15),
        (r"\b(i am|i'm|i work)\b", 0.1),
        (r"\b(update|status|blocked|shipped|launched|finished)\b", 0.15),
        (r"\b(remember|important|note that|keep in mind)\b", 0.2),
    )
)


def heuristic_importance(text: str, base: float = 0.2) -> float:
    """Estimate how worth remembering a piece of text is, from cue phrases alone.

    This is the kind of cheap write-time signal a production memory system can compute without
    an LLM call. It never looks at downstream queries.
    """
    score = base + sum(weight for pattern, weight in _IMPORTANCE_CUES if pattern.search(text))
    return min(1.0, score)
