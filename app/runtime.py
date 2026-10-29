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


def selector_loop_factory() -> asyncio.AbstractEventLoop:
    """Return a selector event loop regardless of platform.

    Uvicorn hardcodes ``ProactorEventLoop`` on Windows, so this factory is
    passed to it via ``--loop app.runtime:selector_loop_factory`` to make
    psycopg's async mode work.
    """
    return asyncio.SelectorEventLoop()
