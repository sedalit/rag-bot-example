"""Shared pytest configuration."""

from app.runtime import configure_event_loop

# psycopg async requires a selector loop on Windows; apply before any
# pytest-asyncio loop is created.
configure_event_loop()
