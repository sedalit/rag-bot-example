"""REST API routes."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_chat_service, get_rate_limiter
from app.api.rate_limit import RateLimiter
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SourceResponse,
)
from app.db.session import get_session
from app.services.chat import ChatService

logger = logging.getLogger(__name__)

router = APIRouter()

SessionDep = Annotated[AsyncSession, Depends(get_session)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
RateLimiterDep = Annotated[RateLimiter, Depends(get_rate_limiter)]


@router.get("/health", response_model=HealthResponse)
async def health(session: SessionDep) -> HealthResponse:
    try:
        await session.execute(text("SELECT 1"))
    except Exception:
        logger.exception("health check failed")
        raise HTTPException(status_code=503, detail="database unavailable") from None
    return HealthResponse(status="ok")


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    chat_service: ChatServiceDep,
    rate_limiter: RateLimiterDep,
) -> ChatResponse:
    if not rate_limiter.allow(payload.session_id):
        raise HTTPException(status_code=429, detail="rate limit exceeded")

    result = await chat_service.answer(payload.session_id, payload.message)

    return ChatResponse(
        answer=result.answer,
        sources=[
            SourceResponse(file=source.file, section=source.section, snippet=source.snippet)
            for source in result.sources
        ],
        latency_ms=result.latency_ms,
    )
