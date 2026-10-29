"""Chat orchestration: basic replies → rewrite → embed → retrieve → threshold → generate."""

import logging
import time
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.ingest.embeddings import Embedder
from app.services.generation import Generator
from app.services.memory import DialogueMemory, Turn
from app.services.retrieval import (
    REFUSAL_ANSWER,
    meets_threshold,
    search_similar,
)
from app.services.rewriting import QueryRewriter
from app.services.smalltalk import match_basic_reply

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
    """Answers questions from retrieved context and handles basic requests.

    Greetings, help requests, thanks and goodbyes are answered deterministically
    before retrieval, so they never reach the embedder or the language model.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        embedder: Embedder,
        generator: Generator,
        rewriter: QueryRewriter,
        memory: DialogueMemory,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._generator = generator
        self._rewriter = rewriter
        self._memory = memory
        self._session_factory = session_factory

    async def answer(self, session_id: str, message: str) -> ChatResult:
        started = time.perf_counter()

        basic = match_basic_reply(message)
        if basic is not None:
            latency_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "basic_reply",
                extra={
                    "session_id": session_id,
                    "intent": basic.intent,
                    "latency_ms": latency_ms,
                },
            )
            # Small talk carries no document context, so it is deliberately not
            # stored in dialogue memory and cannot pollute query rewriting.
            return ChatResult(answer=basic.text, sources=[], latency_ms=latency_ms)

        history = self._memory.get(session_id)
        query = await self._rewriter.rewrite(history, message)

        query_embedding = (await self._embedder.embed([query]))[0]

        search_started = time.perf_counter()
        async with self._session_factory() as session:
            chunks = await search_similar(session, query_embedding, self._settings.top_k)
        search_ms = (time.perf_counter() - search_started) * 1000

        if not meets_threshold(chunks, self._settings.relevance_threshold):
            latency_ms = int((time.perf_counter() - started) * 1000)
            self._log_refusal(session_id, query, search_ms, len(chunks))
            self._remember(session_id, message, REFUSAL_ANSWER)
            return ChatResult(
                answer=REFUSAL_ANSWER, sources=[], latency_ms=latency_ms, refused=True
            )

        # Keep only chunks that clear the threshold so irrelevant context is not
        # injected into the prompt or shown as sources. The search result is
        # sorted by similarity, so the survivors are a non-empty prefix.
        relevant_chunks = [
            chunk for chunk in chunks if chunk.similarity >= self._settings.relevance_threshold
        ]

        generation_started = time.perf_counter()
        generated = await self._generator.generate(query, relevant_chunks)
        generation_ms = (time.perf_counter() - generation_started) * 1000

        sources = [
            Source(file=chunk.file_path, section=chunk.section, snippet=chunk.content)
            for chunk in relevant_chunks
        ]
        latency_ms = int((time.perf_counter() - started) * 1000)

        logger.info(
            "answered",
            extra={
                "session_id": session_id,
                "query": query,
                "rewrote": bool(history),
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
        self._remember(session_id, message, generated.text)
        return ChatResult(answer=generated.text, sources=sources, latency_ms=latency_ms)

    def reset(self, session_id: str) -> None:
        """Clear the dialogue history for a session."""
        self._memory.reset(session_id)

    def _remember(self, session_id: str, user_message: str, assistant_answer: str) -> None:
        self._memory.add(session_id, Turn(role="user", text=user_message))
        self._memory.add(session_id, Turn(role="assistant", text=assistant_answer))

    def _log_refusal(self, session_id: str, query: str, search_ms: float, chunk_count: int) -> None:
        logger.info(
            "refused",
            extra={
                "session_id": session_id,
                "search_ms": round(search_ms, 2),
                "chunks": chunk_count,
                "query": query,
            },
        )
