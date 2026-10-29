FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Use bytecode compilation and copy (not hardlink) from the uv cache.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies and the project in a cached layer.
COPY pyproject.toml uv.lock README.md ./
COPY app ./app
COPY eval ./eval
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Make the venv executables available on PATH.
ENV PATH="/app/.venv/bin:$PATH"

# Copy runtime data and Alembic config.
COPY data ./data
COPY alembic.ini ./

CMD ["python", "-m", "app"]
