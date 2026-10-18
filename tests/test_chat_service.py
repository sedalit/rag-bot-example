"""Unit tests for the chat service orchestration."""

from app.config import get_settings
from app.services.chat import ChatService
from app.services.generation import GeneratedAnswer
from app.services.retrieval import REFUSAL_ANSWER, RetrievedChunk


class _FakeEmbedder:
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return self._vectors


class _FakeGenerator:
    def __init__(self, answer: str, total_tokens: int) -> None:
        self._answer = answer
        self._total_tokens = total_tokens
        self.calls: list[tuple[str, list[RetrievedChunk]]] = []

    async def generate(self, question: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        self.calls.append((question, chunks))
        return GeneratedAnswer(text=self._answer, total_tokens=self._total_tokens)


class _StubSession:
    async def __aenter__(self) -> "_StubSession":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _StubSessionFactory:
    def __call__(self) -> _StubSession:
        return _StubSession()


def _chunk(similarity: float) -> RetrievedChunk:
    return RetrievedChunk(
        content="content",
        section="Section",
        file_path="doc.md",
        chunk_index=0,
        similarity=similarity,
    )


def _service(monkeypatch, chunks: list[RetrievedChunk]) -> ChatService:
    service = ChatService(
        settings=get_settings(),
        embedder=_FakeEmbedder([[0.1, 0.2, 0.3]]),
        generator=_FakeGenerator("the answer", total_tokens=7),
        session_factory=_StubSessionFactory(),
    )

    async def fake_search(session: object, query_embedding: list[float], top_k: int):
        return chunks

    monkeypatch.setattr("app.services.chat.search_similar", fake_search)
    return service


async def test_refuses_below_threshold(monkeypatch):
    result = await _service(monkeypatch, [_chunk(0.1)]).answer("sid", "hello")
    assert result.refused is True
    assert result.answer == REFUSAL_ANSWER
    assert result.sources == []


async def test_answers_above_threshold(monkeypatch):
    result = await _service(monkeypatch, [_chunk(0.5), _chunk(0.4)]).answer("sid", "hello")
    assert result.refused is False
    assert result.answer == "the answer"
    assert [s.file for s in result.sources] == ["doc.md", "doc.md"]
    assert result.latency_ms >= 0
