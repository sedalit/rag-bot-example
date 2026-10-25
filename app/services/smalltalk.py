"""Deterministic replies for greetings and other basic requests."""

import re
from dataclasses import dataclass

GREETING_TEXT = (
    "Здравствуйте! Я DocsBot — отвечаю на вопросы по документации проекта.\n"
    "Например: «Как запустить DocsBot?» или «Как загрузить документы?».\n"
    "Команда /help покажет, что я умею, а /reset очистит историю диалога."
)
HELP_TEXT = (
    "Я отвечаю на вопросы по документации проекта — только по базе знаний "
    "и со ссылками на источники.\n\n"
    "Что можно спросить:\n"
    "• «Как запустить DocsBot?»\n"
    "• «Как загрузить документы?»\n"
    "• «Что возвращает POST /chat?»\n"
    "• «Как сменить провайдера LLM?»\n\n"
    "Команды:\n"
    "/start — приветствие\n"
    "/help — эта справка\n"
    "/reset — очистить историю диалога\n\n"
    "Если ответа в базе нет, я честно об этом скажу."
)
IDENTITY_TEXT = (
    "Я DocsBot — ассистент по документации проекта. Отвечаю на вопросы, "
    "опираясь только на базу знаний, и указываю источники. Могу помочь "
    "с запуском, настройкой, API и архитектурой."
)
THANKS_TEXT = "Пожалуйста! Если появятся ещё вопросы по документации — спрашивайте."
GOODBYE_TEXT = "До свидания! Если понадобится помощь с документацией — возвращайтесь."
SMALLTALK_TEXT = "У меня всё хорошо — я всегда готов помочь с документацией. Что вас интересует?"
ACK_TEXT = "Хорошо! Если появятся вопросы по документации — спрашивайте."


@dataclass(frozen=True)
class BasicReply:
    """A deterministic reply that needs neither retrieval nor generation."""

    intent: str
    text: str


_GREETING_PREFIX = re.compile(
    r"^(?:приветствую|приветик|привет|здравствуйте|здравствуй|"
    r"добрый день|добрый вечер|доброе утро|доброго дня|доброго времени суток|"
    r"хай|салют)(?:\s+|$)"
)
_ADDRESS_PREFIX = re.compile(r"^(?:docsbot|доксбот|бот)(?:\s+|$)")
_ADDRESS_SUFFIX = re.compile(r"(?:^|\s)(?:docsbot|доксбот|бот)$")
_POLITE_PREFIX = re.compile(
    r"^(?:пожалуйста|подскажите|подскажи|скажите|скажи|"
    r"расскажите|расскажи)(?:\s+|$)"
)
_PREFIXES: tuple[tuple[re.Pattern[str], bool], ...] = (
    (_GREETING_PREFIX, True),
    (_ADDRESS_PREFIX, False),
    (_POLITE_PREFIX, False),
)

_CAPABILITIES = re.compile(
    r"^(?:"
    r"что (?:ты )?умеешь(?: делать)?|"
    r"что (?:вы )?умеете(?: делать)?|"
    r"что умеет|"
    r"что (?:ты )?можешь(?: сделать)?|"
    r"что (?:вы )?можете(?: сделать)?|"
    r"что может|"
    r"что ты знаешь|"
    r"чем (?:ты )?можешь помочь|"
    r"чем (?:вы )?можете помочь|"
    r"чем помочь|"
    r"какие (?:у тебя|у вас) возможности|"
    r"твои возможности|ваши возможности|"
    r"как (?:тобой|вами) пользоваться|"
    r"как (?:тебя|вас|бота) использовать|"
    r"как использовать (?:тебя|вас|бота)|"
    r"как пользоваться ботом|"
    r"что можно спросить|"
    r"какие вопросы можно задать|"
    r"помощь|помоги(?:те)?|help"
    r")$"
)
_IDENTITY = re.compile(
    r"^(?:"
    r"кто (?:ты|вы)(?: такой| такая| такое| такие)?|"
    r"кто такой|"
    r"ты кто(?: такой)?|"
    r"вы кто(?: такие)?|"
    r"ты бот|вы бот|ты человек|вы человек|"
    r"что (?:ты|вы) (?:такое|за бот|за программа|за ассистент)|"
    r"что (?:ты|вы)|"
    r"как (?:тебя|вас) зовут|"
    r"(?:твое|ваше) имя|"
    r"о себе|про себя"
    r")$"
)
_THANKS = re.compile(
    r"^(?:"
    r"спасибо(?: тебе| вам)?(?: большое| огромное)?(?: за (?:помощь|ответ))?|"
    r"благодарю(?: (?:тебя|вас))?(?: за (?:помощь|ответ))?|"
    r"спс|мерси"
    r")$"
)
_GOODBYE = re.compile(
    r"^(?:пока|до свидания|до встречи|до скорого|до скорой встречи|прощай(?:те)?|"
    r"всего (?:доброго|хорошего)|хорошего (?:дня|вечера)|увидимся)$"
)
_SMALLTALK = re.compile(
    r"^(?:как (?:дела|ты|вы|настроение|жизнь|поживаешь|"
    r"у тебя дела|у вас дела|ваши дела)|что нового)$"
)
_ACK = re.compile(r"^(?:ок|окей|хорошо|понял|поняла|ясно|принято|ладно)$")

_INTENTS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (_CAPABILITIES, "capabilities", HELP_TEXT),
    (_IDENTITY, "identity", IDENTITY_TEXT),
    (_THANKS, "thanks", THANKS_TEXT),
    (_GOODBYE, "goodbye", GOODBYE_TEXT),
    (_SMALLTALK, "smalltalk", SMALLTALK_TEXT),
    (_ACK, "ack", ACK_TEXT),
)


def _normalize(message: str) -> str:
    """Lowercase and strip punctuation/emoji so matching stays deterministic."""
    text = message.casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _strip_prefixes(text: str) -> tuple[str, bool]:
    """Remove greeting, address and politeness prefixes from a normalized message."""
    saw_greeting = False
    for _ in range(4):
        for pattern, is_greeting in _PREFIXES:
            match = pattern.match(text)
            if match is not None:
                text = text[match.end() :].strip()
                saw_greeting = saw_greeting or is_greeting
                break
        else:
            break
    return text, saw_greeting


def _match_intents(text: str) -> BasicReply | None:
    """Match a fully normalized request against the basic intent patterns."""
    for pattern, intent, reply in _INTENTS:
        if pattern.fullmatch(text):
            return BasicReply(intent, reply)
    return None


def match_basic_reply(message: str) -> BasicReply | None:
    """Return a deterministic reply when the whole message is a basic request."""
    normalized = _normalize(message)
    if not normalized:
        return None

    text, saw_greeting = _strip_prefixes(normalized)
    if not text:
        if saw_greeting:
            return BasicReply("greeting", GREETING_TEXT)
        return None

    reply = _match_intents(text)
    if reply is not None:
        return reply

    # "ты бот" must match before "бот" is treated as a trailing address.
    without_address = _ADDRESS_SUFFIX.sub("", text).strip()
    if without_address != text:
        return _match_intents(without_address)
    return None
