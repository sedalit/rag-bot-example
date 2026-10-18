"""Query rewriting: turn a follow-up message into a standalone query."""

import logging
from pathlib import Path

from openai import AsyncOpenAI

from app.config import Settings
from app.services.memory import Turn

logger = logging.getLogger(__name__)

REWRITE_PROMPT_FILENAME = "rewrite.md"


def load_rewrite_prompt(prompts_dir: Path) -> str:
    """Read the rewrite prompt from its file (kept out of code)."""
    return (prompts_dir / REWRITE_PROMPT_FILENAME).read_text(encoding="utf-8").strip()


def build_rewrite_prompt(history: list[Turn], message: str) -> str:
    """Format the dialogue history plus the current message for the rewriter."""
    lines = []
    for turn in history:
        speaker = "User" if turn.role == "user" else "Assistant"
        lines.append(f"{speaker}: {turn.text}")
    lines.append(f"User: {message}")
    return "\n".join(lines)


class QueryRewriter:
    """Rewrites a follow-up message into a standalone query via an LLM."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        prompt: str,
        temperature: float,
        timeout: float,
    ) -> None:
        self._client = AsyncOpenAI(
            base_url=base_url, api_key=api_key or "not-required", timeout=timeout
        )
        self._model = model
        self._prompt = prompt
        self._temperature = temperature

    @classmethod
    def from_settings(cls, settings: Settings, prompt: str) -> "QueryRewriter":
        """Build a rewriter from application settings and a rewrite prompt."""
        return cls(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.chat_model,
            prompt=prompt,
            temperature=0.0,
            timeout=settings.llm_timeout_seconds,
        )

    async def rewrite(self, history: list[Turn], message: str) -> str:
        """Return a standalone version of ``message`` given the history."""
        if not history:
            return message
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": build_rewrite_prompt(history, message)},
            ],
            temperature=self._temperature,
        )
        rewritten = (response.choices[0].message.content or "").strip()
        logger.info("rewrote query into %d chars", len(rewritten))
        return rewritten or message
