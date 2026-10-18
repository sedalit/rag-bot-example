"""FastAPI application factory and entry point."""

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.rate_limit import RateLimiter
from app.api.routes import router
from app.config import get_settings
from app.db.session import async_session_factory
from app.ingest.embeddings import EmbeddingClient
from app.logging import configure_logging
from app.services.chat import ChatService
from app.services.generation import Generator, load_system_prompt


def create_app() -> FastAPI:
    """Build the application with its service dependencies."""
    settings = get_settings()
    configure_logging(settings.log_level)

    system_prompt = load_system_prompt(settings.prompts_dir)
    embedder = EmbeddingClient.from_settings(settings)
    generator = Generator.from_settings(settings, system_prompt)
    chat_service = ChatService(
        settings=settings,
        embedder=embedder,
        generator=generator,
        session_factory=async_session_factory,
    )
    rate_limiter = RateLimiter(settings.rate_limit_requests, settings.rate_limit_window_seconds)

    app = FastAPI(title=settings.app_name)
    app.state.chat_service = chat_service
    app.state.rate_limiter = rate_limiter

    register_exception_handlers(app)
    app.include_router(router)
    return app


app = create_app()
