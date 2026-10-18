"""Pydantic request/response schemas for the REST API."""

from pydantic import BaseModel, Field, field_validator

from app.config import get_settings


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1)

    @field_validator("message")
    @classmethod
    def message_within_limit(cls, value: str) -> str:
        limit = get_settings().max_message_length
        if len(value) > limit:
            raise ValueError(f"message is too long (max {limit} characters)")
        return value


class SourceResponse(BaseModel):
    file: str
    section: str
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    latency_ms: int


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str
