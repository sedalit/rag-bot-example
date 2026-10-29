"""FastAPI dependency providers."""

from typing import cast

from fastapi import Request

from app.api.rate_limit import RateLimiter
from app.services.chat import ChatService


def get_chat_service(request: Request) -> ChatService:
    return cast(ChatService, request.app.state.chat_service)


def get_rate_limiter(request: Request) -> RateLimiter:
    return cast(RateLimiter, request.app.state.rate_limiter)
