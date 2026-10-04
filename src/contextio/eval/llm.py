"""Optional end-to-end evaluation: answer queries with a model and grade the answers."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Protocol, runtime_checkable

from contextio.bench.dataset import BenchQuery, BenchUser
from contextio.selection import Selection

SYSTEM_PROMPT = (
    "You are a long-term personal assistant. Use the notes about the user when they are "
    "relevant, prefer the most recent information when notes conflict, and answer concisely."
)

PROMPT_TEMPLATE = """\
Notes about the user (oldest first within each section):
{context}

User request: {query}
"""


@runtime_checkable
class ChatModel(Protocol):
    def complete(self, prompt: str, system: str | None = None) -> str: ...


class OpenAICompatibleClient:
    """Minimal client for any OpenAI-compatible ``/chat/completions`` endpoint.

    Works with OpenAI, OpenRouter, vLLM, Ollama (``base_url="http://localhost:11434/v1"``), and
    other servers that implement the same API. Uses only the standard library.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        max_tokens: int = 300,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, prompt: str, system: str | None = None) -> str:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read())
        return body["choices"][0]["message"]["content"] or ""


def build_prompt(query: BenchQuery, selection: Selection) -> str:
    context = selection.render() or "(no notes)"
    return PROMPT_TEMPLATE.format(context=context, query=query.text)


def keyword_grade(answer: str, keywords: tuple[str, ...]) -> bool | None:
    """True when every keyword appears in the answer (case-insensitive); None if ungradable."""
    if not keywords:
        return None
    lowered = answer.lower()
    return all(k.lower() in lowered for k in keywords)


class KeywordAnswerer:
    """Answers each query from the selected context and grades it by its answer keywords.

    Queries without keywords (such as style requests) are skipped and recorded as ungraded, so
    they cost nothing. Pass an instance as ``answerer`` to ``run_benchmark``.
    """

    def __init__(self, model: ChatModel, system: str = SYSTEM_PROMPT) -> None:
        self.model = model
        self.system = system
        self.calls = 0

    def __call__(self, user: BenchUser, query: BenchQuery, selection: Selection) -> bool | None:
        if not query.answer_keywords:
            return None
        self.calls += 1
        answer = self.model.complete(build_prompt(query, selection), system=self.system)
        return keyword_grade(answer, query.answer_keywords)
