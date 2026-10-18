"""Unit tests for the batched embedding client with retries."""

import pytest

from app.ingest.embeddings import EmbeddingClient


class _Item:
    def __init__(self, vector: list[float]) -> None:
        self.embedding = vector


class _Response:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.data = [_Item(v) for v in vectors]


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.responses: list[list[list[float]]] = []

    async def create(self, *, model: str, input: list[str], **kwargs: object) -> _Response:
        self.calls.append(list(input))
        return _Response(self.responses.pop(0))


class _FlakyEmbeddings:
    """Fails the first ``failures`` calls, then succeeds."""

    def __init__(self, failures: int) -> None:
        self.attempts = 0
        self.failures = failures

    async def create(self, *, model: str, input: list[str], **kwargs: object) -> _Response:
        self.attempts += 1
        if self.attempts <= self.failures:
            raise RuntimeError("transient provider error")
        return _Response([[1.0, 1.0, 1.0] for _ in input])


class _FakeOpenAI:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


async def _no_sleep(_seconds: float) -> None:
    return None


def _make_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    embeddings: object,
    batch_size: int = 2,
    max_retries: int = 2,
    expected_dim: int = 3,
) -> tuple[EmbeddingClient, object]:
    captured: dict[str, object] = {}

    class OpenAIWithEmbeddings:
        def __init__(self, **kwargs: object) -> None:
            self.embeddings = embeddings

    monkeypatch.setattr("app.ingest.embeddings.AsyncOpenAI", OpenAIWithEmbeddings)
    monkeypatch.setattr("app.ingest.embeddings.asyncio.sleep", _no_sleep)

    client = EmbeddingClient(
        base_url="http://test",
        api_key="key",
        model="test-model",
        batch_size=batch_size,
        max_retries=max_retries,
        expected_dim=expected_dim,
        timeout=1.0,
    )
    captured["client"] = client
    return client, captured


async def test_embed_splits_into_batches_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeEmbeddings()
    fake.responses = [
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        [[0.7, 0.8, 0.9]],
    ]
    client, _ = _make_client(monkeypatch, embeddings=fake, batch_size=2)

    result = await client.embed(["a", "b", "c"])

    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    assert fake.calls == [["a", "b"], ["c"]]


async def test_embed_empty_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _make_client(monkeypatch, embeddings=_FakeEmbeddings())
    assert await client.embed([]) == []


async def test_embed_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FlakyEmbeddings(failures=2)
    client, _ = _make_client(monkeypatch, embeddings=fake, max_retries=3)

    result = await client.embed(["a"])

    assert result == [[1.0, 1.0, 1.0]]
    assert fake.attempts == 3


async def test_embed_raises_after_retries_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FlakyEmbeddings(failures=999)
    client, _ = _make_client(monkeypatch, embeddings=fake, max_retries=3)

    with pytest.raises(RuntimeError, match="transient provider error"):
        await client.embed(["a"])

    assert fake.attempts == 3


async def test_embed_validates_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeEmbeddings()
    fake.responses = [[[0.1, 0.2]]]  # wrong dimension (2 instead of 3)
    client, _ = _make_client(monkeypatch, embeddings=fake, expected_dim=3)

    with pytest.raises(ValueError, match="dimension"):
        await client.embed(["a"])
