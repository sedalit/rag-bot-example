"""Cross-platform asyncio configuration helpers."""

import asyncio
import sys


def configure_event_loop() -> None:
    """Select a psycopg-compatible event loop on Windows.

    psycopg's async mode does not support the ``ProactorEventLoop`` that
    ``asyncio`` uses by default on Windows, so we switch to the selector loop
    before any entry point creates the event loop.
    """
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
