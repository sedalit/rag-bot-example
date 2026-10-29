"""Unit tests for query rewriting."""

from app.services.memory import Turn
from app.services.rewriting import QueryRewriter, build_rewrite_prompt


def test_build_rewrite_prompt_formats_history():
    history = [
        Turn(role="user", text="How do I load documents?"),
        Turn(role="assistant", text="Run the ingest command"),
    ]
    prompt = build_rewrite_prompt(history, "And the chunk size?")
    assert "Пользователь: How do I load documents?" in prompt
    assert "Ассистент: Run the ingest command" in prompt
    assert prompt.endswith("Пользователь: And the chunk size?")


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    async def create(self, **kwargs: object) -> _Response:
        return _Response("standalone question")


class _Chat:
    def __init__(self) -> None:
        self.completions = _Completions()


class _OpenAI:
    def __init__(self, **kwargs: object) -> None:
        self.chat = _Chat()


def _rewriter() -> QueryRewriter:
    return QueryRewriter(
        base_url="http://test", api_key="k", model="m", prompt="p", temperature=0.0, timeout=1.0
    )


async def test_rewrite_empty_history_returns_message_unchanged():
    assert await _rewriter().rewrite([], "hello") == "hello"


async def test_rewrite_with_history_uses_llm(monkeypatch):
    monkeypatch.setattr("app.services.client.AsyncOpenAI", _OpenAI)
    history = [Turn(role="user", text="How do I load documents?")]
    result = await _rewriter().rewrite(history, "And the chunk size?")
    assert result == "standalone question"
