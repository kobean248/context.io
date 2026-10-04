"""Memory consolidation: turn raw conversation history into compact, deduplicated facts.

    raw conversations -> extraction -> facts -> deduplication -> storage -> retrieval

Consolidated memories keep ``source_ids`` pointing at the raw memories they came from, so they
can be evaluated against the same ground truth.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from contextio.memory import Memory, MemoryKind, MemoryStore, heuristic_importance
from contextio.retrieval.base import Query, StoreCache
from contextio.selection import Selection
from contextio.strategies import Strategy
from contextio.text import jaccard, tokenize

Completer = Callable[[str], str]

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_PRONOUNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in (
        (r"\bI'm\b", "User is"),
        (r"\bI am\b", "User is"),
        (r"\bI've\b", "User has"),
        (r"\bI have\b", "User has"),
        (r"\bI'd\b", "User would"),
        (r"\bI'll\b", "User will"),
        (r"\bI\b", "User"),
        (r"\bmy\b", "their"),
        (r"\bmine\b", "theirs"),
        (r"\bme\b", "them"),
        (r"\bwe've\b", "the team has"),
        (r"\bwe're\b", "the team is"),
        (r"\bwe\b", "the team"),
        (r"\bour\b", "their"),
        (r"\bus\b", "them"),
    )
)

_THIRD_PERSON_VERBS = frozenset(
    "use prefer work write like want need think love hate build run enjoy find plan "
    "lean stick keep".split()
)
_USER_VERB_RE = re.compile(r"\bUser ((?:always|never|usually|mostly|now|still|really) )?(\w+)\b")


def to_third_person(sentence: str) -> str:
    """Rewrite a first-person statement as a fact about the user."""
    text = sentence.strip()
    for pattern, replacement in _PRONOUNS:
        text = pattern.sub(replacement, text)

    def conjugate(match: re.Match[str]) -> str:
        adverb, verb = match.group(1) or "", match.group(2)
        if verb.lower() in _THIRD_PERSON_VERBS:
            verb += "s"
        return f"User {adverb}{verb}"

    text = _USER_VERB_RE.sub(conjugate, text)
    return text[:1].upper() + text[1:]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def deduplicate(memories: Sequence[Memory], threshold: float = 0.8) -> list[Memory]:
    """Merge near-duplicate memories (token-set Jaccard >= ``threshold``).

    The newest memory of each duplicate group is kept, with the group's source ids merged and
    the highest importance retained.
    """
    kept: list[Memory] = []
    kept_terms: list[set[str]] = []
    for memory in sorted(memories, key=lambda m: (-m.timestamp, m.id)):
        terms = set(tokenize(memory.text))
        for i, other_terms in enumerate(kept_terms):
            if jaccard(terms, other_terms) >= threshold:
                survivor = kept[i]
                sources = dict.fromkeys((*survivor.source_ids, *memory.covers))
                sources.pop(survivor.id, None)
                kept[i] = Memory(
                    id=survivor.id,
                    text=survivor.text,
                    kind=survivor.kind,
                    timestamp=survivor.timestamp,
                    importance=max(survivor.importance, memory.importance),
                    tags=survivor.tags | memory.tags,
                    source_ids=tuple(sources),
                )
                break
        else:
            kept.append(memory)
            kept_terms.append(terms)
    return sorted(kept, key=lambda m: (m.timestamp, m.id))


@runtime_checkable
class Consolidator(Protocol):
    def consolidate(self, memories: Sequence[Memory]) -> list[Memory]: ...


class RuleBasedConsolidator:
    """Extract fact-like sentences from episodic memories without calling a model.

    A sentence is fact-like when it contains a preference, decision, change, or status cue (see
    ``heuristic_importance``). Each one is rewritten in the third person as a SEMANTIC memory.
    Static and already-semantic memories pass through unchanged; episodic memories with no
    fact-like sentences are dropped unless ``keep_unmatched`` is set.
    """

    def __init__(self, keep_unmatched: bool = False, dedup_threshold: float = 0.8) -> None:
        self.keep_unmatched = keep_unmatched
        self.dedup_threshold = dedup_threshold

    def consolidate(self, memories: Sequence[Memory]) -> list[Memory]:
        output: list[Memory] = []
        facts: list[Memory] = []
        for memory in memories:
            if memory.kind is not MemoryKind.EPISODIC:
                output.append(memory)
                continue
            extracted = self._extract(memory)
            facts.extend(extracted)
            if not extracted and self.keep_unmatched:
                output.append(memory)
        output.extend(deduplicate(facts, self.dedup_threshold))
        return output

    @staticmethod
    def _extract(memory: Memory) -> list[Memory]:
        facts = []
        for sentence in split_sentences(memory.text):
            importance = heuristic_importance(sentence)
            if importance <= heuristic_importance(""):
                continue
            facts.append(
                Memory(
                    id=f"{memory.id}#f{len(facts)}",
                    text=to_third_person(sentence),
                    kind=MemoryKind.SEMANTIC,
                    timestamp=memory.timestamp,
                    importance=importance,
                    tags=memory.tags,
                    source_ids=tuple(sorted(memory.covers)),
                )
            )
        return facts


_LLM_PROMPT = """\
You maintain long-term memory for an AI assistant. Extract durable facts about the user from
the conversation snippets below: preferences, background, projects, decisions, and the latest
status of ongoing work. When a newer snippet contradicts an older one, keep only the newer fact.
Skip small talk and generic explanations.

Respond with only a JSON array. Each element must be an object with keys:
  "fact": one short sentence in the third person ("User prefers ..."),
  "source_ids": ids of the snippets that support the fact,
  "importance": number from 0 to 1.

Snippets:
{snippets}
"""


class LLMConsolidator:
    """Ask a language model to extract and reconcile facts, in batches of episodic memories.

    ``complete`` is any function that sends a prompt to a model and returns its text response,
    which keeps this class independent of any provider SDK.
    """

    def __init__(self, complete: Completer, batch_size: int = 40) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.complete = complete
        self.batch_size = batch_size

    def consolidate(self, memories: Sequence[Memory]) -> list[Memory]:
        passthrough = [m for m in memories if m.kind is not MemoryKind.EPISODIC]
        episodic = sorted(
            (m for m in memories if m.kind is MemoryKind.EPISODIC),
            key=lambda m: (m.timestamp, m.id),
        )
        by_id = {m.id: m for m in episodic}
        facts: list[Memory] = []
        for start in range(0, len(episodic), self.batch_size):
            batch = episodic[start : start + self.batch_size]
            snippets = "\n".join(f"[{m.id}] (t={m.timestamp:g}) {m.text}" for m in batch)
            response = self.complete(_LLM_PROMPT.format(snippets=snippets))
            for item in _parse_json_array(response):
                sources = [s for s in item.get("source_ids", []) if s in by_id]
                text = str(item.get("fact", "")).strip()
                if not text or not sources:
                    continue
                importance = item.get("importance", 0.5)
                facts.append(
                    Memory(
                        id=f"fact-{start // self.batch_size}-{len(facts)}",
                        text=text,
                        kind=MemoryKind.SEMANTIC,
                        timestamp=max(by_id[s].timestamp for s in sources),
                        importance=min(1.0, max(0.0, float(importance))),
                        source_ids=tuple(sources),
                    )
                )
        return passthrough + deduplicate(facts)


def _parse_json_array(response: str) -> list[dict]:
    start, end = response.find("["), response.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(response[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


class CompressedStrategy:
    """Run ``inner`` over a consolidated copy of the user's memory store.

    Consolidation runs once per store and is cached, mirroring a system that consolidates memory
    offline rather than on the request path.
    """

    def __init__(
        self, inner: Strategy, consolidator: Consolidator | None = None, name: str | None = None
    ) -> None:
        self.inner = inner
        self.consolidator = consolidator or RuleBasedConsolidator()
        self.name = name or f"compressed-{inner.name}"
        self._cache: StoreCache[MemoryStore] = StoreCache(self._consolidate)

    def _consolidate(self, store: MemoryStore) -> MemoryStore:
        return MemoryStore(self.consolidator.consolidate(list(store)), store.token_counter)

    def consolidated(self, store: MemoryStore) -> MemoryStore:
        return self._cache.get(store)

    def select(self, query: Query, store: MemoryStore, budget: int | None) -> Selection:
        return self.inner.select(query, self.consolidated(store), budget)
