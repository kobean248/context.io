"""Benchmark data model: users with memory histories and labeled queries."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from contextio.memory import Memory, MemoryStore
from contextio.retrieval.base import Query
from contextio.tokens import TokenCounter


@dataclass(frozen=True)
class BenchQuery:
    id: str
    text: str
    kind: str
    required_ids: frozenset[str]
    """Memories that must be in context to answer correctly."""
    stale_ids: frozenset[str] = field(default_factory=frozenset)
    """Superseded memories that would lead to an outdated answer."""
    answer_keywords: tuple[str, ...] = ()
    """Strings a correct answer must mention; empty when the answer cannot be graded by keyword."""
    timestamp: float | None = None

    def to_query(self) -> Query:
        return Query(self.text, timestamp=self.timestamp)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "kind": self.kind,
            "required_ids": sorted(self.required_ids),
            "stale_ids": sorted(self.stale_ids),
            "answer_keywords": list(self.answer_keywords),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BenchQuery:
        return cls(
            id=data["id"],
            text=data["text"],
            kind=data["kind"],
            required_ids=frozenset(data["required_ids"]),
            stale_ids=frozenset(data.get("stale_ids", ())),
            answer_keywords=tuple(data.get("answer_keywords", ())),
            timestamp=data.get("timestamp"),
        )


@dataclass
class BenchUser:
    id: str
    memories: list[Memory]
    queries: list[BenchQuery]

    def store(self, token_counter: TokenCounter | None = None) -> MemoryStore:
        return MemoryStore(self.memories, token_counter)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "memories": [m.to_dict() for m in self.memories],
            "queries": [q.to_dict() for q in self.queries],
        }

    @classmethod
    def from_dict(cls, data: dict) -> BenchUser:
        return cls(
            id=data["id"],
            memories=[Memory.from_dict(m) for m in data["memories"]],
            queries=[BenchQuery.from_dict(q) for q in data["queries"]],
        )


@dataclass
class Dataset:
    users: list[BenchUser]
    name: str = "dataset"

    def __iter__(self) -> Iterator[BenchUser]:
        return iter(self.users)

    def __len__(self) -> int:
        return len(self.users)

    @property
    def n_queries(self) -> int:
        return sum(len(u.queries) for u in self.users)

    def stats(self, token_counter: TokenCounter | None = None) -> dict:
        stores = [u.store(token_counter) for u in self.users]
        tokens = [s.total_tokens for s in stores]
        return {
            "name": self.name,
            "users": len(self.users),
            "memories": sum(len(s) for s in stores),
            "queries": self.n_queries,
            "avg_tokens_per_user": sum(tokens) / len(tokens) if tokens else 0.0,
            "max_tokens_per_user": max(tokens, default=0),
            "query_kinds": dict(Counter(q.kind for u in self.users for q in u.queries)),
        }

    def save_jsonl(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as f:
            for user in self.users:
                f.write(json.dumps(user.to_dict()) + "\n")

    @classmethod
    def load_jsonl(cls, path: str | Path) -> Dataset:
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            users = [BenchUser.from_dict(json.loads(line)) for line in f if line.strip()]
        return cls(users, name=path.stem)
