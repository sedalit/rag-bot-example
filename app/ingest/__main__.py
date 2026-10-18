"""CLI entry point: ``python -m app.ingest --path ./data/docs``."""

import argparse
import asyncio
import logging
from pathlib import Path

from app.config import get_settings
from app.db.session import async_session_factory, engine
from app.ingest.chunking import MarkdownChunker
from app.ingest.embeddings import EmbeddingClient
from app.ingest.ingest import ingest
from app.runtime import configure_event_loop


def main() -> None:
    configure_event_loop()
    parser = argparse.ArgumentParser(description="Ingest Markdown documents into the vector store.")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("data/docs"),
        help="Directory containing .md files (default: data/docs).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    settings = get_settings()
    chunker = MarkdownChunker(settings.chunk_size, settings.chunk_overlap)
    embedder = EmbeddingClient.from_settings(settings)

    async def run() -> None:
        try:
            stats = await ingest(args.path, async_session_factory, embedder, chunker)
            print(
                f"added={stats.added} updated={stats.updated} removed={stats.removed} "
                f"skipped={stats.skipped} chunks={stats.chunks}"
            )
        finally:
            await engine.dispose()

    asyncio.run(run())


if __name__ == "__main__":
    main()
