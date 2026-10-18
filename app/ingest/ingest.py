"""Idempotent document ingestion orchestration."""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Chunk as ChunkModel
from app.db.models import Document
from app.ingest.chunking import Chunk, MarkdownChunker
from app.ingest.embeddings import Embedder

logger = logging.getLogger(__name__)


def hash_file(content: bytes) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    return hashlib.sha256(content).hexdigest()


def scan_markdown_files(docs_dir: Path) -> dict[str, bytes]:
    """Return ``{relative_posix_path: raw_bytes}`` for every ``*.md`` file."""
    if not docs_dir.is_dir():
        raise NotADirectoryError(f"not a directory: {docs_dir}")
    return {
        file.relative_to(docs_dir).as_posix(): file.read_bytes()
        for file in sorted(docs_dir.rglob("*.md"))
    }


def plan_ingestion(
    existing: dict[str, str],
    current: dict[str, str],
) -> tuple[set[str], set[str], set[str]]:
    """Compute ``(to_add, to_update, to_delete)`` from stored and on-disk hashes.

    ``existing`` and ``current`` map a relative file path to its content hash.
    A file is re-embedded only when its hash changed; files no longer on disk
    are removed.
    """
    existing_paths = set(existing)
    current_paths = set(current)
    to_add = {path for path in current_paths if path not in existing_paths}
    to_update = {path for path in current_paths & existing_paths if existing[path] != current[path]}
    to_delete = existing_paths - current_paths
    return to_add, to_update, to_delete


@dataclass
class IngestStats:
    """Summary counters returned after an ingestion run."""

    added: int = 0
    updated: int = 0
    removed: int = 0
    skipped: int = 0
    chunks: int = 0
    paths: list[str] = field(default_factory=list)


async def ingest(
    docs_dir: Path,
    session_factory: async_sessionmaker[AsyncSession],
    embedder: Embedder,
    chunker: MarkdownChunker,
) -> IngestStats:
    """Ingest all Markdown files under ``docs_dir`` idempotently."""
    files = scan_markdown_files(docs_dir)
    current_hashes = {path: hash_file(data) for path, data in files.items()}

    async with session_factory() as session:
        rows = (await session.execute(select(Document))).scalars().all()
    existing_hashes = {doc.path: doc.content_hash for doc in rows}

    to_add, to_update, to_delete = plan_ingestion(existing_hashes, current_hashes)

    stats = IngestStats()
    stats.removed = len(to_delete)
    stats.skipped = len(current_hashes) - len(to_add) - len(to_update)

    for path in sorted(to_add | to_update):
        await _upsert_document(
            path=path,
            raw=files[path],
            is_new=path in to_add,
            chunker=chunker,
            embedder=embedder,
            session_factory=session_factory,
            stats=stats,
        )

    if to_delete:
        await _delete_documents(to_delete, session_factory)

    return stats


async def _upsert_document(
    *,
    path: str,
    raw: bytes,
    is_new: bool,
    chunker: MarkdownChunker,
    embedder: Embedder,
    session_factory: async_sessionmaker[AsyncSession],
    stats: IngestStats,
) -> None:
    text = raw.decode("utf-8")
    chunks: list[Chunk] = chunker.chunk(text)
    embeddings = await embedder.embed([chunk.content for chunk in chunks])
    if len(embeddings) != len(chunks):
        raise ValueError(
            f"embedder returned {len(embeddings)} vectors for {len(chunks)} chunks of {path}"
        )

    content_hash = hash_file(raw)
    async with session_factory() as session:
        document = (
            await session.execute(select(Document).where(Document.path == path))
        ).scalar_one_or_none()

        if document is None:
            document = Document(path=path, content_hash=content_hash)
            session.add(document)
            await session.flush()  # assign document.id before referencing it
        else:
            document.content_hash = content_hash
            await session.execute(delete(ChunkModel).where(ChunkModel.document_id == document.id))

        for chunk, vector in zip(chunks, embeddings, strict=True):
            session.add(
                ChunkModel(
                    document_id=document.id,
                    chunk_index=chunk.index,
                    section=chunk.section,
                    content=chunk.content,
                    embedding=vector,
                )
            )

        await session.commit()

    if is_new:
        stats.added += 1
    else:
        stats.updated += 1
    stats.chunks += len(chunks)
    stats.paths.append(path)
    logger.info("ingested %s (%d chunks)", path, len(chunks))


async def _delete_documents(
    paths: set[str],
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        # Chunks are removed by the DB-level ON DELETE CASCADE foreign key.
        await session.execute(delete(Document).where(Document.path.in_(paths)))
        await session.commit()
    for path in sorted(paths):
        logger.info("removed %s", path)
