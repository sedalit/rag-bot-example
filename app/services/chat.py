"""Chat orchestration: embed → retrieve → threshold → generate."""

import logging
import time
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.ingest.embeddings import Embedder
from app.services.generation import Generator
from app.services.retrieval import (
    REFUSAL_ANSWER,
    meets_threshold,
    search_similar,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Source:
    """A cited source: file, section and matching snippet."""

    file: str
    section: str
    snippet: str


@dataclass(frozen=True)
class ChatResult:
    """The outcome of answering one message."""

    answer: str
    sources: list[Source]
    latency_ms: int
    refused: bool = False


class ChatService:
    """Answers questions by retrieving context and generating a grounded reply."""

    def __init__(
        self,
        *,
        settings: Settings,
        embedder: Embedder,
        generator: Generator,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._generator = generator
        self._session_factory = session_factory

    async def answer(self, session_id: str, message: str) -> ChatResult:
        started = time.perf_counter()

        query_embedding = (await self._embedder.embed([message]))[0]

        search_started = time.perf_counter()
        async with self._session_factory() as session:
            chunks = await search_similar(session, query_embedding, self._settings.top_k)
        search_ms = (time.perf_counter() - search_started) * 1000

        if not meets_threshold(chunks, self._settings.relevance_threshold):
            latency_ms = int((time.perf_counter() - started) * 1000)
            self._log_refusal(session_id, message, search_ms, len(chunks))
            return ChatResult(
                answer=REFUSAL_ANSWER, sources=[], latency_ms=latency_ms, refused=True
            )

        generation_started = time.perf_counter()
        generated = await self._generator.generate(message, chunks)
        generation_ms = (time.perf_counter() - generation_started) * 1000

        sources = [
            Source(file=chunk.file_path, section=chunk.section, snippet=chunk.content)
            for chunk in chunks
        ]
        latency_ms = int((time.perf_counter() - started) * 1000)

        logger.info(
            "answered",
            extra={
                "session_id": session_id,
                "search_ms": round(search_ms, 2),
                "generation_ms": round(generation_ms, 2),
                "total_tokens": generated.total_tokens,
                "chunks": [
                    {
                        "file": chunk.file_path,
                        "section": chunk.section,
                        "similarity": round(chunk.similarity, 4),
                    }
                    for chunk in chunks
                ],
            },
        )
        return ChatResult(answer=generated.text, sources=sources, latency_ms=latency_ms)

    def _log_refusal(
        self, session_id: str, message: str, search_ms: float, chunk_count: int
    ) -> None:
        logger.info(
            "refused",
            extra={
                "session_id": session_id,
                "search_ms": round(search_ms, 2),
                "chunks": chunk_count,
                "query": message,
            },
        )
