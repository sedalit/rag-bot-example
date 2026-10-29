"""Unit tests for the Telegram bot reply formatting."""

from app.bot.bot import UNKNOWN_COMMAND_TEXT, WELCOME, format_reply
from app.services.chat import Source
from app.services.smalltalk import HELP_TEXT


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


def test_welcome_mentions_help_and_reset():
    assert "DocsBot" in WELCOME
    assert "/help" in WELCOME
    assert "/reset" in WELCOME


def test_help_and_unknown_command_list_available_commands():
    for command in ("/start", "/help", "/reset"):
        assert command in HELP_TEXT
        assert command in UNKNOWN_COMMAND_TEXT
