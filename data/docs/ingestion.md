# Ingestion

Ingestion is the pipeline that loads Markdown documentation into the vector
store so it can be retrieved later. It is run from the command line and is
designed to be idempotent.

## Chunking

Each Markdown file is first split into sections by its headings. Nested
headings form a heading path, for example `Configuration > Chunking`. Each
section is then cut into chunks of at most the configured chunk size, with a
small overlap between adjacent chunks so context is not lost at the boundary.

Every chunk keeps metadata: the file path, the section heading, and its order
within the document. This metadata is returned to the user as source
references, so an answer can be traced back to a specific section.

## Embeddings

Chunk text is embedded in batches through the configured OpenAI-compatible
endpoint. Failed requests are retried with exponential backoff up to a
configured maximum. Embeddings are validated against the configured dimension
before they are written to PostgreSQL.

## Idempotency

Each file's content is hashed with SHA-256. On every run the pipeline compares
stored hashes against the files on disk. Files whose hash is unchanged are
skipped entirely, so embeddings are never recomputed for them. Files that
changed are re-chunked and re-embedded, replacing their old chunks. Files that
were deleted from disk have their documents and chunks removed from the
database.

## Running

Run the pipeline with:

```
python -m app.ingest --path ./data/docs
```

The command prints a summary of added, updated, removed, and skipped files
plus the total number of chunks written.
