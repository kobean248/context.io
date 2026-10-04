import pytest

from contextio.memory import Memory, MemoryKind, MemoryStore


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(
        [
            Memory(id="name", text="The user's name is Priya.", kind=MemoryKind.STATIC),
            Memory(
                id="lang",
                text="Primary programming languages: Go and Rust.",
                kind=MemoryKind.STATIC,
            ),
            Memory(id="db-old", text="I mostly use MySQL for my side projects.", timestamp=10),
            Memory(
                id="db-new",
                text="I've switched to PostgreSQL for every database now.",
                timestamp=50,
                importance=0.8,
            ),
            Memory(id="weather", text="It was raining all weekend, so I stayed in.", timestamp=60),
            Memory(
                id="cache",
                text="For the ledger project we decided to use Redis for caching.",
                timestamp=40,
                importance=0.9,
            ),
            Memory(
                id="filler",
                text="Caching layers trade memory for latency; Memcached and Redis are common "
                "choices, and cache invalidation remains one of the hard problems.",
                timestamp=41,
            ),
        ]
    )
