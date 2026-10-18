# API Reference

The REST API exposes two endpoints: a health check and a chat endpoint.

## Health check

`GET /health` reports whether the service and its database are available. It
returns HTTP 200 when healthy and HTTP 503 when the database cannot be
reached. It is intended for load balancers and container orchestration.

## Chat

`POST /chat` accepts a JSON body with a `session_id` string and a `message`
string. Messages are validated and length-limited. The response contains an
`answer` string, a `sources` array, and a `latency_ms` integer.

Each source object identifies a `file`, a `section`, and the matching
`snippet` of chunk text. The answer text references sources with bracketed
numbers such as [1] that correspond to the order of the sources array.

## Errors

All errors use a consistent JSON shape with an `error` object carrying a
`code` and a human-readable `message`. Validation failures return HTTP 422,
unknown routes return 404, and requests exceeding the per-session rate limit
return 429.

## Refusal behaviour

When no retrieved chunk meets the relevance threshold, the endpoint does not
call the language model. It returns a fixed refusal message stating that the
knowledge base contains no information on the question.
