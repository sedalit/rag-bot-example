"""FastAPI application factory and entry point."""

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.rate_limit import RateLimiter
from app.api.routes import router
from app.config import get_settings
from app.logging import configure_logging
from app.services.factory import build_chat_service


def create_app() -> FastAPI:
    """Build the application with its service dependencies."""
    settings = get_settings()
    configure_logging(settings.log_level)

    chat_service = build_chat_service(settings)
    rate_limiter = RateLimiter(settings.rate_limit_requests, settings.rate_limit_window_seconds)

    app = FastAPI(title=settings.app_name)
    app.state.chat_service = chat_service
    app.state.rate_limiter = rate_limiter

    register_exception_handlers(app)
    app.include_router(router)
    return app


app = create_app()
