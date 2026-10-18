# Getting Started

DocsBot is a retrieval-augmented generation (RAG) service that answers
questions strictly from a Markdown knowledge base. Every answer is generated
from retrieved chunks only and always cites its sources. When the knowledge
base has no relevant material, the bot honestly says it does not know.

## Prerequisites

To run DocsBot you need Docker with Docker Compose. The stack bundles a
PostgreSQL 16 database with the pgvector extension, a FastAPI application, and
an optional Telegram bot. All services are described in a single
`docker-compose.yml` file at the repository root.

## Quick start

Start the whole stack with one command:

```
docker compose up
```

The API listens on port 8000 by default. The database listens on port 5432.
The Telegram bot starts only when `TELEGRAM_ENABLED=true` and a valid
`TELEGRAM_BOT_TOKEN` are present in the environment.

## Loading documents

Before asking questions, load the Markdown files from the documentation
directory into the vector store:

```
python -m app.ingest --path ./data/docs
```

The command splits every Markdown file into chunks by headings, computes
embeddings in batches, and stores them in PostgreSQL. Re-running the command
is safe: unchanged files are skipped by hash, changed files are re-embedded,
and deleted files are removed from the database.

## Asking a question

Send a chat request to the REST API:

```
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "message": "How do I load documents?"}'
```

The response contains the generated answer, a list of source references, and
the measured latency in milliseconds.
