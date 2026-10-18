"""Run the Telegram bot: ``python -m app.bot``."""

import asyncio

from aiogram import Bot

from app.bot.bot import build_dispatcher
from app.config import get_settings
from app.db.session import engine
from app.logging import configure_logging
from app.runtime import configure_event_loop
from app.services.factory import build_chat_service


def main() -> None:
    configure_event_loop()
    settings = get_settings()
    if not settings.telegram_enabled or not settings.telegram_bot_token:
        print("Telegram bot is disabled (TELEGRAM_ENABLED=false or TELEGRAM_BOT_TOKEN is empty).")
        return

    configure_logging(settings.log_level)
    chat_service = build_chat_service(settings)

    async def run() -> None:
        bot = Bot(token=settings.telegram_bot_token)
        dispatcher = build_dispatcher(chat_service)
        try:
            await dispatcher.start_polling(bot)
        finally:
            await engine.dispose()

    asyncio.run(run())


if __name__ == "__main__":
    main()
