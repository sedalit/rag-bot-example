"""Telegram bot handlers and dispatcher factory."""

from aiogram import Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message

from app.services.chat import ChatService, Source

WELCOME = (
    "Hi! I'm DocsBot. Ask me anything about the project documentation.\n"
    "Use /reset to clear our conversation history."
)


def format_reply(answer: str, sources: list[Source]) -> str:
    """Append a source list to the answer for display."""
    if not sources:
        return answer
    lines = [f"- {source.file} — {source.section}" for source in sources]
    return f"{answer}\n\nSources:\n" + "\n".join(lines)


def build_dispatcher(chat_service: ChatService) -> Dispatcher:
    """Build a dispatcher whose handlers reuse the shared ChatService."""
    dp = Dispatcher()

    @dp.message(Command("start"))
    async def start(message: Message) -> None:
        await message.answer(WELCOME)

    @dp.message(Command("reset"))
    async def reset(message: Message) -> None:
        chat_service.reset(str(message.chat.id))
        await message.answer("Conversation history cleared.")

    @dp.message(F.text)
    async def chat(message: Message) -> None:
        text = message.text or ""
        result = await chat_service.answer(str(message.chat.id), text)
        await message.answer(format_reply(result.answer, result.sources))

    return dp
