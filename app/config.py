"""Application configuration via environment variables (.env)."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root, used to resolve default relative paths regardless of cwd.
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Typed application settings, loaded from the environment and a local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application ---------------------------------------------------------
    app_name: str = "DocsBot"
    environment: str = "development"
    log_level: str = "INFO"

    # Database ------------------------------------------------------------
    # psycopg3 works for both sync (Alembic) and async (runtime) engines.
    # 127.0.0.1 avoids the ::1 IPv6 loopback that can hang on Docker Desktop.
    database_url: str = "postgresql+psycopg://docsbot:docsbot@127.0.0.1:5432/docsbot"
    # Dimension of the embedding column; must match the configured embedding
    # model. Changing it requires a new Alembic migration (see app/db/models.py).
    embedding_dim: int = 1536

    # OpenAI-compatible provider -----------------------------------------
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 60.0

    # Ingestion -----------------------------------------------------------
    chunk_size: int = 800
    chunk_overlap: int = 100
    embedding_batch_size: int = 32
    embedding_max_retries: int = 3

    # Retrieval -----------------------------------------------------------
    top_k: int = 5
    relevance_threshold: float = 0.30

    # Dialogue memory -----------------------------------------------------
    memory_size: int = 6

    # REST API ------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_message_length: int = 2000
    rate_limit_requests: int = 10
    rate_limit_window_seconds: int = 60

    # Prompts -------------------------------------------------------------
    prompts_dir: Path = Field(default_factory=lambda: BASE_DIR / "app" / "prompts")

    # Telegram bot --------------------------------------------------------
    telegram_enabled: bool = False
    telegram_bot_token: str = ""


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()
