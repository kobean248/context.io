"""Lightweight text normalization shared by lexical and embedding-based components."""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+(?:[+#][a-z0-9+#]*)?|[a-z0-9]*[+#]+")

STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers herself him himself his how i if in into is it its itself
    just me more most my myself no nor not now of off on once only or other our ours ourselves out
    over own same she should so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what when where which
    while who whom why will with would you your yours yourself yourselves let lets s t ll ve re d m
    also really actually like get got going go yeah ok okay thing things
    """.split()
)


def words(text: str) -> list[str]:
    """Lowercased word tokens, keeping tokens such as ``c++`` and ``c#`` intact."""
    return _WORD_RE.findall(text.lower())


def tokenize(text: str) -> list[str]:
    """Content-bearing word tokens: lowercased, stopwords removed, light plural stemming."""
    return [_stem(w) for w in words(text) if w not in STOPWORDS]


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def char_ngrams(word: str, n: int = 3) -> list[str]:
    padded = f"<{word}>"
    if len(padded) <= n:
        return [padded]
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)
