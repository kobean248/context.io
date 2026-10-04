import math

import pytest

from contextio.memory import Memory, MemoryStore
from contextio.retrieval import (
    Embedder,
    HashingEmbedder,
    Query,
    Retriever,
    VectorRetriever,
    cosine,
)


def test_hashing_embedder_is_deterministic_and_normalized():
    embedder = HashingEmbedder(dim=64)
    assert isinstance(embedder, Embedder)
    a1, a2 = embedder.embed(["deploying services", "deploying services"])
    assert a1 == a2
    assert len(a1) == 64
    assert math.sqrt(sum(x * x for x in a1)) == pytest.approx(1.0)


def test_hashing_embedder_handles_empty_text():
    (vector,) = HashingEmbedder(dim=8).embed([""])
    assert vector == [0.0] * 8


def test_char_ngrams_give_morphological_overlap():
    embedder = HashingEmbedder()
    deploy, deployment, banana = embedder.embed(["deploy", "deployment", "banana"])
    assert cosine(deploy, deployment) > cosine(deploy, banana)


def test_hashing_embedder_validates_dim():
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)


def test_vector_retriever_ranks_semantically_closest_first(store):
    retriever = VectorRetriever()
    assert isinstance(retriever, Retriever)
    scored = retriever.score(Query("what programming languages do I write?"), store)
    assert scored[0].memory.id == "lang"


def test_vector_retriever_uses_custom_embedder_and_caches_memories():
    calls = []

    class CountingEmbedder:
        def embed(self, texts):
            calls.append(list(texts))
            return [[1.0, 0.0] if "cat" in t else [0.0, 1.0] for t in texts]

    store = MemoryStore([Memory(id="c", text="a cat"), Memory(id="d", text="a dog")])
    retriever = VectorRetriever(CountingEmbedder())
    assert retriever.score(Query("cat"), store)[0].memory.id == "c"
    assert retriever.score(Query("dog"), store)[0].memory.id == "d"
    assert sum(len(c) for c in calls) == 4


def test_vector_retriever_empty_store():
    assert VectorRetriever().score(Query("x"), MemoryStore()) == []
