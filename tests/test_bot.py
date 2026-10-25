"""Unit tests for the Telegram bot reply formatting."""

from app.bot.bot import format_reply
from app.services.chat import Source


def test_format_reply_without_sources_returns_answer():
    assert format_reply("the answer", []) == "the answer"


def test_format_reply_appends_sources():
    sources = [
        Source(file="a.md", section="Section A", snippet="snippet a"),
        Source(file="b.md", section="Section B", snippet="snippet b"),
    ]
    reply = format_reply("the answer", sources)
    assert reply.startswith("the answer\n\nИсточники:\n")
    assert "- a.md — Section A" in reply
    assert "- b.md — Section B" in reply
