"""Integration tests for the REST API with a mocked chat service."""

import httpx
import pytest

from app.api.rate_limit import RateLimiter
from app.db.session import get_session
from app.main import create_app
from app.services.chat import ChatResult, Source


class _FakeChatService:
    async def answer(self, session_id: str, message: str) -> ChatResult:
        return ChatResult(
            answer="hello",
            sources=[Source(file="doc.md", section="Section", snippet="snippet")],
            latency_ms=12,
        )


class _FakeSession:
    def __init__(self, ok: bool = True) -> None:
        self._ok = ok

    async def execute(self, *args: object, **kwargs: object) -> object:
        if not self._ok:
            raise RuntimeError("database unavailable")
        return None


@pytest.fixture
def app():
    application = create_app()
    application.state.chat_service = _FakeChatService()
    application.state.rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
    return application


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_chat_returns_answer_and_sources(client):
    resp = await client.post("/chat", json={"session_id": "s1", "message": "hi"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "hello"
    assert data["sources"] == [{"file": "doc.md", "section": "Section", "snippet": "snippet"}]
    assert isinstance(data["latency_ms"], int)


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
        yield _FakeSession(ok=True)

    app.dependency_overrides[get_session] = ok_session
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_health_unavailable(app, client):
    async def bad_session():
        yield _FakeSession(ok=False)

    app.dependency_overrides[get_session] = bad_session
    resp = await client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "http_error"
