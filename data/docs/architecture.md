# Architecture

DocsBot follows a layered architecture. Business logic lives in dedicated
service modules and does not depend on FastAPI or aiogram, which keeps the
core testable and reusable across the REST API and the Telegram interface.

## Layers

The `api/` layer exposes HTTP routes. The `bot/` layer implements the Telegram
interface. Both call the same service layer, so the answering logic is written
once and never duplicated between channels.

The `services/` layer contains three responsibilities: retrieval, generation,
and dialogue memory. Retrieval turns a query into an embedding and fetches the
top-k most similar chunks from pgvector. Generation builds a prompt from the
retrieved chunks and asks the language model for a grounded answer. Memory
keeps a short history of recent turns per session.

## Data flow

A question flows through several stages. First the message is validated and
rate-limited. Then the query is embedded and matched against stored chunks by
cosine similarity. If the best match is below the relevance threshold, the
service returns a refusal without calling the model. Otherwise the chunks are
formatted into a context block and the model generates an answer with inline
citations such as [1] and [2].

## Storage

PostgreSQL 16 stores documents and their chunks. The `documents` table tracks
each source file path and its content hash for idempotent ingestion. The
`chunks` table holds chunk text, section metadata, and a `vector` column
indexed with an HNSW cosine index. Embeddings have a fixed dimension that must
match the configured embedding model.

## Extensibility

The LLM and embedding endpoints are OpenAI-compatible, so the provider can be
changed by editing environment variables without touching code. Future
improvements such as hybrid search, a reranker, or streaming responses fit
into the same service boundaries.
