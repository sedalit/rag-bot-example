# Configuration

DocsBot reads all settings from environment variables, with a template provided
in `.env.example`. Secrets such as the LLM API key must never be committed to
the repository.

## Database

`DATABASE_URL` points to PostgreSQL using the psycopg dialect. Use the IPv4
loopback address `127.0.0.1` rather than `localhost` to avoid IPv6 resolution
issues under Docker Desktop. The `EMBEDDING_DIM` value must match the output
dimension of the configured embedding model.

## Provider

The language model and embeddings are configured through an OpenAI-compatible
API. `LLM_BASE_URL` sets the endpoint, `LLM_API_KEY` the secret key, and
`CHAT_MODEL` and `EMBEDDING_MODEL` select the concrete models. This lets you
swap providers, for example from the hosted OpenAI API to a self-hosted
endpoint, without any code changes.

## Chunking

`CHUNK_SIZE` and `CHUNK_OVERLAP` control how documents are split. The default
chunk size is 800 characters with a 100 character overlap. Larger chunks carry
more context but can dilute retrieval precision; smaller chunks are more
precise but may lose surrounding context.

## Retrieval

`TOP_K` controls how many chunks are retrieved per query, defaulting to five.
`RELEVANCE_THRESHOLD` is the minimum cosine similarity for the best chunk;
queries below it are refused. `MEMORY_SIZE` is the number of recent dialogue
turns kept per session.

## Rate limiting

The API applies a simple in-memory rate limit per session. `RATE_LIMIT_REQUESTS`
and `RATE_LIMIT_WINDOW_SECONDS` define how many chat requests a session may
send within the window before receiving a 429 response.

## Telegram

Set `TELEGRAM_ENABLED=true` and provide `TELEGRAM_BOT_TOKEN` to start the bot.
When disabled, the bot process exits immediately and only the API runs.
