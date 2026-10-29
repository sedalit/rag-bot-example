"""Unit tests for the chat service orchestration."""

from app.config import get_settings
from app.services.chat import ChatService
from app.services.generation import GeneratedAnswer
from app.services.memory import DialogueMemory, Turn
from app.services.retrieval import REFUSAL_ANSWER, RetrievedChunk
from app.services.smalltalk import GREETING_TEXT


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


class _FakeRewriter:
    def __init__(self, rewritten: str | None = None) -> None:
        self._rewritten = rewritten
        self.calls: list[tuple[list[Turn], str]] = []

    async def rewrite(self, history: list[Turn], message: str) -> str:
        self.calls.append((history, message))
        return self._rewritten or message


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


def _service(
    monkeypatch,
    chunks: list[RetrievedChunk],
    *,
    rewriter: _FakeRewriter | None = None,
    memory: DialogueMemory | None = None,
) -> ChatService:
    service = ChatService(
        settings=get_settings(),
        embedder=_FakeEmbedder([[0.1, 0.2, 0.3]]),
        generator=_FakeGenerator("the answer", total_tokens=7),
        rewriter=rewriter or _FakeRewriter(),
        memory=memory or DialogueMemory(max_turns=6),
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


async def test_rewrites_followup_when_history_exists(monkeypatch):
    memory = DialogueMemory(max_turns=6)
    memory.add("sid", Turn(role="user", text="How do I load documents?"))
    memory.add("sid", Turn(role="assistant", text="Run the ingest command"))
    rewriter = _FakeRewriter(rewritten="What is the default chunk size?")
    service = _service(monkeypatch, [_chunk(0.5)], rewriter=rewriter, memory=memory)

    await service.answer("sid", "And the chunk size?")

    assert rewriter.calls
    history, message = rewriter.calls[0]
    assert len(history) == 2
    assert message == "And the chunk size?"


async def test_generates_from_rewritten_query(monkeypatch):
    memory = DialogueMemory(max_turns=6)
    memory.add("sid", Turn(role="user", text="How do I load documents?"))
    memory.add("sid", Turn(role="assistant", text="Run the ingest command"))
    rewriter = _FakeRewriter(rewritten="How do I load documents, standalone")
    generator = _FakeGenerator("the answer", total_tokens=7)
    service = ChatService(
        settings=get_settings(),
        embedder=_FakeEmbedder([[0.1, 0.2, 0.3]]),
        generator=generator,
        rewriter=rewriter,
        memory=memory,
        session_factory=_StubSessionFactory(),
    )

    async def fake_search(session: object, query_embedding: list[float], top_k: int):
        return [_chunk(0.5)]

    monkeypatch.setattr("app.services.chat.search_similar", fake_search)

    await service.answer("sid", "And the chunk size?")

    assert generator.calls[0][0] == "How do I load documents, standalone"


async def test_filters_sub_threshold_chunks_before_generation(monkeypatch):
    generator = _FakeGenerator("the answer", total_tokens=7)
    service = ChatService(
        settings=get_settings(),
        embedder=_FakeEmbedder([[0.1, 0.2, 0.3]]),
        generator=generator,
        rewriter=_FakeRewriter(),
        memory=DialogueMemory(max_turns=6),
        session_factory=_StubSessionFactory(),
    )

    async def fake_search(session: object, query_embedding: list[float], top_k: int):
        return [_chunk(0.5), _chunk(0.1), _chunk(0.05)]

    monkeypatch.setattr("app.services.chat.search_similar", fake_search)

    result = await service.answer("sid", "hello")

    assert [s.file for s in result.sources] == ["doc.md"]
    assert len(generator.calls[0][1]) == 1
    assert generator.calls[0][1][0].similarity == 0.5


async def test_reset_clears_history(monkeypatch):
    memory = DialogueMemory(max_turns=6)
    service = _service(monkeypatch, [_chunk(0.5)], memory=memory)
    await service.answer("sid", "hello")
    assert memory.get("sid")

    service.reset("sid")
    assert memory.get("sid") == []


async def test_basic_request_bypasses_retrieval_and_generation(monkeypatch):
    memory = DialogueMemory(max_turns=6)
    rewriter = _FakeRewriter()
    generator = _FakeGenerator("the answer", total_tokens=7)
    service = ChatService(
        settings=get_settings(),
        embedder=_FakeEmbedder([[0.1, 0.2, 0.3]]),
        generator=generator,
        rewriter=rewriter,
        memory=memory,
        session_factory=_StubSessionFactory(),
    )

    async def fail_search(session: object, query_embedding: list[float], top_k: int):
        raise AssertionError("search must not run for a basic request")

    monkeypatch.setattr("app.services.chat.search_similar", fail_search)

    result = await service.answer("sid", "Привет!")

    assert result.answer == GREETING_TEXT
    assert result.sources == []
    assert result.refused is False
    assert rewriter.calls == []
    assert generator.calls == []
    assert memory.get("sid") == []
