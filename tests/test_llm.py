import io
import json

import pytest

from contextio.bench import BenchQuery, generate_dataset
from contextio.eval import BenchmarkConfig, run_benchmark, summarize
from contextio.eval.llm import (
    ChatModel,
    KeywordAnswerer,
    OpenAICompatibleClient,
    build_prompt,
    keyword_grade,
)
from contextio.memory import Memory, MemoryKind
from contextio.selection import make_selection
from contextio.strategies import FullContext


class EchoModel:
    """Answers with the context it was given, so grading depends only on what was selected."""

    def complete(self, prompt: str, system: str | None = None) -> str:
        return prompt


def test_keyword_grade():
    assert keyword_grade("Use PostgreSQL.", ("postgresql",)) is True
    assert keyword_grade("Use MySQL.", ("PostgreSQL",)) is False
    assert keyword_grade("anything", ()) is None


def test_build_prompt_includes_context_and_query():
    query = BenchQuery(id="q", text="Which DB?", kind="preference", required_ids=frozenset())
    selection = make_selection(
        [Memory(id="a", text="Prefers PostgreSQL.", kind=MemoryKind.STATIC)], lambda m: 3, None
    )
    prompt = build_prompt(query, selection)
    assert "- Prefers PostgreSQL." in prompt
    assert "User request: Which DB?" in prompt
    assert "(no notes)" in build_prompt(query, make_selection([], lambda m: 0, None))


def test_keyword_answerer_in_benchmark():
    dataset = generate_dataset(n_users=2, seed=3)
    answerer = KeywordAnswerer(EchoModel())
    assert isinstance(EchoModel(), ChatModel)
    records = run_benchmark(
        dataset, [FullContext()], BenchmarkConfig(budgets=(1000,)), answerer=answerer
    )
    graded = [r for r in records if r.correct is not None]
    ungraded = [r for r in records if r.correct is None]
    assert answerer.calls == len(graded)
    assert all(r.query_kind == "style" for r in ungraded)
    assert all(r.correct for r in graded)
    assert summarize(records)[0].accuracy == 1.0


def test_openai_compatible_client_builds_request(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        response = {"choices": [{"message": {"content": "Use PostgreSQL."}}]}
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleClient(
        "test-model", base_url="http://localhost:8000/v1/", api_key="secret", timeout=5
    )
    assert client.complete("Which DB?", system="be brief") == "Use PostgreSQL."
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["body"]["model"] == "test-model"
    assert captured["body"]["messages"][0] == {"role": "system", "content": "be brief"}
    assert captured["timeout"] == 5


def test_client_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert OpenAICompatibleClient("m").api_key == "from-env"
    monkeypatch.delenv("OPENAI_API_KEY")
    assert OpenAICompatibleClient("m").api_key == ""


@pytest.mark.parametrize("keywords", [("a",), ("a", "b")])
def test_keyword_grade_requires_all(keywords):
    assert keyword_grade("a b", keywords) is True
