"""Unit tests for deterministic basic replies."""

import pytest

from app.services.smalltalk import (
    ACK_TEXT,
    GOODBYE_TEXT,
    GREETING_TEXT,
    HELP_TEXT,
    IDENTITY_TEXT,
    SMALLTALK_TEXT,
    THANKS_TEXT,
    match_basic_reply,
)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Привет!", GREETING_TEXT),
        ("Здравствуйте, DocsBot!", GREETING_TEXT),
        ("Доброе утро", GREETING_TEXT),
        ("Приветствую", GREETING_TEXT),
        ("Привет, привет", GREETING_TEXT),
        ("DocsBot, привет", GREETING_TEXT),
        ("Что ты умеешь?", HELP_TEXT),
        ("Что умеешь делать?", HELP_TEXT),
        ("Что вы умеете?", HELP_TEXT),
        ("Что умеет DocsBot?", HELP_TEXT),
        ("Чем вы можете помочь?", HELP_TEXT),
        ("Какие у вас возможности?", HELP_TEXT),
        ("Как вами пользоваться?", HELP_TEXT),
        ("Добрый день, что ты умеешь?", HELP_TEXT),
        ("Подскажи, пожалуйста, чем ты можешь помочь", HELP_TEXT),
        ("Помощь", HELP_TEXT),
        ("help", HELP_TEXT),
        ("Кто ты?", IDENTITY_TEXT),
        ("Кто такой DocsBot?", IDENTITY_TEXT),
        ("Ты бот?", IDENTITY_TEXT),
        ("Что вы за бот?", IDENTITY_TEXT),
        ("Как тебя зовут?", IDENTITY_TEXT),
        ("Расскажи о себе", IDENTITY_TEXT),
        ("Спасибо большое!", THANKS_TEXT),
        ("Спасибо за помощь!", THANKS_TEXT),
        ("Спасибо вам большое!", THANKS_TEXT),
        ("Благодарю вас", THANKS_TEXT),
        ("Пока!", GOODBYE_TEXT),
        ("До свидания", GOODBYE_TEXT),
        ("До скорой встречи", GOODBYE_TEXT),
        ("Как дела?", SMALLTALK_TEXT),
        ("Как у тебя дела, бот?", SMALLTALK_TEXT),
        ("Как вы?", SMALLTALK_TEXT),
        ("Что нового?", SMALLTALK_TEXT),
        ("Ок", ACK_TEXT),
        ("Хорошо!", ACK_TEXT),
    ],
)
def test_matches_basic_requests(message: str, expected: str) -> None:
    reply = match_basic_reply(message)
    assert reply is not None
    assert reply.text == expected


@pytest.mark.parametrize(
    "message",
    [
        "Привет! Как загрузить документы?",
        "Что ты умеешь настраивать?",
        "Что вы умеете настраивать?",
        "Что умеет эта система с эмбеддингами?",
        "Спасибо, а как загрузить документы?",
        "Спасибо за помощь с загрузкой",
        "Спасибо вам за помощь с загрузкой",
        "Расскажи про архитектуру проекта",
        "Кто ты в этом проекте?",
        "Кто такой Пушкин?",
        "Ты бот или человек?",
        "help me please",
        "",
        "   ",
    ],
)
def test_does_not_match_questions(message: str) -> None:
    assert match_basic_reply(message) is None


def test_reports_intent_for_logging() -> None:
    reply = match_basic_reply("Привет!")
    assert reply is not None
    assert reply.intent == "greeting"
