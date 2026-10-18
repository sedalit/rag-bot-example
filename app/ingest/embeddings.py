"""Embedding client for an OpenAI-compatible API, with batching and retries."""

import asyncio
import logging
from collections.abc import Sequence
from typing import Protocol

from openai import AsyncOpenAI

from app.config import Settings

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Minimal interface any embedding backend must satisfy."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


def _batches(texts: Sequence[str], size: int) -> list[list[str]]:
    return [list(texts[i : i + size]) for i in range(0, len(texts), size)]


class EmbeddingClient:
    """Embeds texts through an OpenAI-compatible embeddings endpoint.

    Requests are split into batches and retried with exponential backoff on
    API errors.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        batch_size: int,
        max_retries: int,
        expected_dim: int,
        timeout: float,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_retries < 1:
            raise ValueError("max_retries must be positive")
        self._model = model
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._expected_dim = expected_dim
        # A placeholder key keeps construction valid for local providers that
        # ignore auth; real requests still send the configured key.
        self._client = AsyncOpenAI(
            base_url=base_url, api_key=api_key or "not-required", timeout=timeout
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "EmbeddingClient":
        """Build a client from application settings."""
        return cls(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            max_retries=settings.embedding_max_retries,
            expected_dim=settings.embedding_dim,
            timeout=settings.llm_timeout_seconds,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed all texts, preserving order, and return one vector per text."""
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for batch in _batches(texts, self._batch_size):
            vectors = await self._embed_with_retries(batch)
            self._validate_dimensions(vectors)
            embeddings.extend(vectors)
        return embeddings

    async def _embed_with_retries(self, batch: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return await self._embed_batch(batch)
            except Exception as exc:  # retry transient/unknown provider errors
                last_error = exc
                if attempt < self._max_retries - 1:
                    wait = 2**attempt
                    logger.warning(
                        "embedding request failed (attempt %d/%d), retrying in %.0fs: %s",
                        attempt + 1,
                        self._max_retries,
                        wait,
                        exc,
                    )
                    await asyncio.sleep(wait)
        assert last_error is not None
        raise last_error

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(model=self._model, input=batch)
        return [item.embedding for item in response.data]

    def _validate_dimensions(self, vectors: list[list[float]]) -> None:
        """Reject vectors whose size does not match the configured dimension.

        Kept outside the retry loop: a dimension mismatch is a configuration
        error, not a transient provider failure.
        """
        for vector in vectors:
            if len(vector) != self._expected_dim:
                raise ValueError(
                    f"embedding dimension {len(vector)} != configured {self._expected_dim}"
                )
