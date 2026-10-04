"""BM25 keyword retrieval."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from contextio.memory import MemoryStore
from contextio.retrieval.base import Query, Scored, StoreCache, rank
from contextio.text import tokenize


@dataclass
class _Index:
    term_freqs: dict[str, Counter[str]]
    lengths: dict[str, int]
    doc_freq: Counter[str]
    avg_length: float


class BM25Retriever:
    """Okapi BM25 over memory text, with the index cached per store."""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("expected k1 >= 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        self._cache: StoreCache[_Index] = StoreCache(self._build)

    @staticmethod
    def _build(store: MemoryStore) -> _Index:
        term_freqs: dict[str, Counter[str]] = {}
        lengths: dict[str, int] = {}
        doc_freq: Counter[str] = Counter()
        for memory in store:
            terms = tokenize(memory.text)
            tf = Counter(terms)
            term_freqs[memory.id] = tf
            lengths[memory.id] = len(terms)
            doc_freq.update(tf.keys())
        avg_length = sum(lengths.values()) / len(lengths) if lengths else 0.0
        return _Index(term_freqs, lengths, doc_freq, avg_length)

    def score(self, query: Query, store: MemoryStore) -> list[Scored]:
        index = self._cache.get(store)
        n_docs = len(store)
        query_terms = set(tokenize(query.text))
        idf = {
            t: math.log(1 + (n_docs - index.doc_freq[t] + 0.5) / (index.doc_freq[t] + 0.5))
            for t in query_terms
            if index.doc_freq[t]
        }

        results = []
        for memory in store:
            tf = index.term_freqs[memory.id]
            norm = self.k1 * (
                1 - self.b + self.b * index.lengths[memory.id] / (index.avg_length or 1.0)
            )
            total = 0.0
            for term, weight in idf.items():
                freq = tf.get(term, 0)
                if freq:
                    total += weight * freq * (self.k1 + 1) / (freq + norm)
            results.append(Scored(memory, total))
        return rank(results)
