"""Shared service construction for the API and the Telegram bot."""

from app.config import Settings
from app.db.session import async_session_factory
from app.ingest.embeddings import EmbeddingClient
from app.services.chat import ChatService
from app.services.generation import Generator, load_system_prompt
from app.services.memory import DialogueMemory
from app.services.rewriting import QueryRewriter, load_rewrite_prompt


def build_chat_service(settings: Settings) -> ChatService:
    """Build a fully wired ChatService so both channels share one implementation."""
    system_prompt = load_system_prompt(settings.prompts_dir)
    rewrite_prompt = load_rewrite_prompt(settings.prompts_dir)
    return ChatService(
        settings=settings,
        embedder=EmbeddingClient.from_settings(settings),
        generator=Generator.from_settings(settings, system_prompt),
        rewriter=QueryRewriter.from_settings(settings, rewrite_prompt),
        memory=DialogueMemory(settings.memory_size),
        session_factory=async_session_factory,
    )
