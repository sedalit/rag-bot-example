"""Integration tests for the REST API with a mocked LLM and database."""

import httpx
import pytest

from app.api.rate_limit import RateLimiter
from app.config import get_settings
from app.db.session import get_session
from app.ingest.embeddings import EmbeddingClient
from app.main import create_app
from app.services.chat import ChatService
from app.services.generation import Generator
from app.services.memory import DialogueMemory
from app.services.retrieval import REFUSAL_ANSWER

# --- Fakes for the OpenAI-compatible clients (no network) --------------------


class _FakeEmbedding:
    def __init__(self, vector: list[float]) -> None:
        self.embedding = vector


class _FakeEmbeddingsResponse:
    def __init__(self, data: list[_FakeEmbedding]) -> None:
        self.data = data


class _FakeEmbeddingsEndpoint:
    async def create(
        self, *, model: str, input: list[str], **kwargs: object
    ) -> _FakeEmbeddingsResponse:
        return _FakeEmbeddingsResponse([_FakeEmbedding([0.1, 0.2, 0.3]) for _ in input])


class _FakeEmbeddingsClient:
    def __init__(self, **kwargs: object) -> None:
        self.embeddings = _FakeEmbeddingsEndpoint()


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self) -> None:
        self.total_tokens = 10


class _FakeChatResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


class _FakeCompletions:
    async def create(self, **kwargs: object) -> _FakeChatResponse:
        return _FakeChatResponse("mocked answer [1]")


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeChatClient:
    def __init__(self, **kwargs: object) -> None:
        self.chat = _FakeChat()


class _IdentityRewriter:
    async def rewrite(self, history: object, message: str) -> str:
        return message


# --- Fakes for the database session used by retrieval ------------------------


class _FakeRow:
    def __init__(
        self, content: str, section: str, file_path: str, chunk_index: int, similarity: float
    ) -> None:
        self.content = content
        self.section = section
        self.file_path = file_path
        self.chunk_index = chunk_index
        self.similarity = similarity


class _FakeSearchResult:
    def __init__(self, rows: list[_FakeRow]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class _FakeSearchSession:
    def __init__(self, rows: list[_FakeRow]) -> None:
        self._rows = rows

    async def execute(self, *args: object, **kwargs: object) -> _FakeSearchResult:
        return _FakeSearchResult(self._rows)


class _FakeSearchSessionCtx:
    def __init__(self, session: _FakeSearchSession) -> None:
        self._session = session

    async def __aenter__(self) -> _FakeSearchSession:
        return self._session

    async def __aexit__(self, *exc: object) -> None:
        return None


class _FakeSearchSessionFactory:
    def __init__(self, rows: list[_FakeRow]) -> None:
        self._rows = rows

    def __call__(self) -> _FakeSearchSessionCtx:
        return _FakeSearchSessionCtx(_FakeSearchSession(self._rows))


# --- Fake session for the /health DB check -----------------------------------


class _HealthSession:
    def __init__(self, ok: bool = True) -> None:
        self._ok = ok

    async def execute(self, *args: object, **kwargs: object) -> object:
        if not self._ok:
            raise RuntimeError("database unavailable")
        return None


_ABOVE_ROWS = [
    _FakeRow("content one", "Section A", "doc.md", 0, 0.8),
    _FakeRow("content two", "Section B", "other.md", 1, 0.6),
]
_BELOW_ROWS = [_FakeRow("irrelevant", "Section", "doc.md", 0, 0.1)]


def _make_chat_service(monkeypatch: pytest.MonkeyPatch, rows: list[_FakeRow]) -> ChatService:
    monkeypatch.setattr("app.ingest.embeddings.AsyncOpenAI", _FakeEmbeddingsClient)
    monkeypatch.setattr("app.services.generation.AsyncOpenAI", _FakeChatClient)

    embedder = EmbeddingClient(
        base_url="http://test",
        api_key="test",
        model="test-model",
        batch_size=2,
        max_retries=1,
        expected_dim=3,
        timeout=1.0,
    )
    generator = Generator(
        base_url="http://test",
        api_key="test",
        model="test-model",
        system_prompt="system",
        temperature=0.0,
        max_tokens=10,
        timeout=1.0,
    )
    return ChatService(
        settings=get_settings(),
        embedder=embedder,
        generator=generator,
        rewriter=_IdentityRewriter(),
        memory=DialogueMemory(max_turns=6),
        session_factory=_FakeSearchSessionFactory(rows),
    )


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch):
    application = create_app()
    application.state.chat_service = _make_chat_service(monkeypatch, _ABOVE_ROWS)
    application.state.rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
    return application


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_chat_runs_real_pipeline_with_mocked_llm(client):
    resp = await client.post("/chat", json={"session_id": "s1", "message": "hi"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "mocked answer [1]"
    assert data["sources"] == [
        {"file": "doc.md", "section": "Section A", "snippet": "content one"},
        {"file": "other.md", "section": "Section B", "snippet": "content two"},
    ]
    assert isinstance(data["latency_ms"], int)


async def test_chat_refuses_below_threshold(app, monkeypatch: pytest.MonkeyPatch):
    app.state.chat_service = _make_chat_service(monkeypatch, _BELOW_ROWS)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post("/chat", json={"session_id": "s1", "message": "hi"})

    assert resp.status_code == 200
    assert resp.json()["answer"] == REFUSAL_ANSWER
    assert resp.json()["sources"] == []


async def test_chat_rejects_empty_message_with_unified_error(client):
    resp = await client.post("/chat", json={"session_id": "s1", "message": ""})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


async def test_chat_rejects_too_long_message(client):
    resp = await client.post("/chat", json={"session_id": "s1", "message": "x" * 2001})
    assert resp.status_code == 422


async def test_chat_rate_limited(app):
    app.state.rate_limiter = RateLimiter(max_requests=1, window_seconds=60)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        first = await c.post("/chat", json={"session_id": "s1", "message": "hi"})
        second = await c.post("/chat", json={"session_id": "s1", "message": "hi"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "http_error"


async def test_health_ok(app, client):
    async def ok_session():
        yield _HealthSession(ok=True)

    app.dependency_overrides[get_session] = ok_session
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_health_unavailable(app, client):
    async def bad_session():
        yield _HealthSession(ok=False)

    app.dependency_overrides[get_session] = bad_session
    resp = await client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "http_error"
