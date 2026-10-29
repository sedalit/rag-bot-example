"""Vector retrieval over stored chunks via pgvector cosine similarity."""

import logging
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Returned when no retrieved chunk meets the relevance threshold.
REFUSAL_ANSWER = "В базе знаний нет информации по этому вопросу."


@dataclass(frozen=True)
class RetrievedChunk:
    """A stored chunk together with its source metadata and similarity."""

    content: str
    section: str
    file_path: str
    chunk_index: int
    similarity: float


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(value) for value in vector) + "]"


async def search_similar(
    session: AsyncSession,
    query_embedding: list[float],
    top_k: int,
) -> list[RetrievedChunk]:
    """Return up to ``top_k`` chunks ordered by cosine similarity (descending)."""
    result = await session.execute(
        text(
            """
            SELECT c.content, c.section, c.chunk_index, d.path AS file_path,
                   1 - (c.embedding <=> CAST(:query AS vector)) AS similarity
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL
            ORDER BY c.embedding <=> CAST(:query AS vector)
            LIMIT :top_k
            """
        ),
        {"query": _vector_literal(query_embedding), "top_k": top_k},
    )
    chunks = [
        RetrievedChunk(
            content=row.content,
            section=row.section,
            file_path=row.file_path,
            chunk_index=row.chunk_index,
            similarity=float(row.similarity),
        )
        for row in result
    ]
    logger.debug("retrieved %d chunks, top similarity %.4f", len(chunks), best_similarity(chunks))
    return chunks


def best_similarity(chunks: list[RetrievedChunk]) -> float:
    """Return the highest similarity among retrieved chunks (0.0 when empty)."""
    return chunks[0].similarity if chunks else 0.0


def meets_threshold(chunks: list[RetrievedChunk], threshold: float) -> bool:
    """Return whether the best chunk clears the relevance threshold."""
    return bool(chunks) and chunks[0].similarity >= threshold
