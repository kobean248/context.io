"""Token counting.

Every budget in contextio is expressed in tokens, so token counting is pluggable. The default
heuristic counter is deterministic and dependency-free, which keeps benchmark results
reproducible; ``TiktokenCounter`` gives exact counts for OpenAI-style BPE vocabularies.
"""

from __future__ import annotations

import math
import re
from typing import Protocol, runtime_checkable

_PIECE_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@runtime_checkable
class TokenCounter(Protocol):
    def __call__(self, text: str) -> int: ...


class HeuristicTokenCounter:
    """Approximates BPE token counts without a vocabulary.

    Text is split into word and punctuation pieces; each piece costs one token per
    ``chars_per_token`` characters (minimum one). Common words therefore map to a single token
    and long identifiers to several, which tracks BPE tokenizers closely on English prose and
    code (roughly 1.3 tokens per word).
    """

    def __init__(self, chars_per_token: float = 6.0) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self.chars_per_token = chars_per_token

    def __call__(self, text: str) -> int:
        return sum(
            max(1, math.ceil(len(p) / self.chars_per_token)) for p in _PIECE_RE.findall(text)
        )

    def __repr__(self) -> str:
        return f"HeuristicTokenCounter(chars_per_token={self.chars_per_token})"


class TiktokenCounter:
    """Exact token counts using ``tiktoken`` (install with ``pip install contextio[tiktoken]``)."""

    def __init__(self, encoding: str = "o200k_base") -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                "TiktokenCounter requires tiktoken: pip install 'contextio[tiktoken]'"
            ) from exc
        self.encoding_name = encoding
        self._encoding = tiktoken.get_encoding(encoding)

    def __call__(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))

    def __repr__(self) -> str:
        return f"TiktokenCounter(encoding={self.encoding_name!r})"


def default_counter() -> TokenCounter:
    return HeuristicTokenCounter()
