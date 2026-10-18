"""Answer generation from retrieved context via an OpenAI-compatible API."""

import logging
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from app.config import Settings
from app.services.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_FILENAME = "system.md"


@dataclass(frozen=True)
class GeneratedAnswer:
    """The model's answer together with its reported token usage."""

    text: str
    total_tokens: int


def _total_tokens(response: ChatCompletion) -> int:
    usage = response.usage
    if usage is not None and usage.total_tokens is not None:
        return usage.total_tokens
    return 0


def load_system_prompt(prompts_dir: Path) -> str:
    """Read the system prompt from its file (kept out of code)."""
    path = prompts_dir / SYSTEM_PROMPT_FILENAME
    return path.read_text(encoding="utf-8").strip()


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as numbered context blocks for the prompt."""
    blocks = [
        f"[{index}] {chunk.file_path} — {chunk.section}\n{chunk.content}"
        for index, chunk in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks)


def build_user_prompt(question: str, context: str) -> str:
    """Assemble the user message from the question and retrieved context."""
    return f"Context:\n{context}\n\nQuestion: {question}"


class Generator:
    """Calls the chat model with a system prompt and retrieved context."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> None:
        # A placeholder key keeps construction valid for local providers that
        # ignore auth; real requests still send the configured key.
        self._client = AsyncOpenAI(
            base_url=base_url, api_key=api_key or "not-required", timeout=timeout
        )
        self._model = model
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens

    @classmethod
    def from_settings(cls, settings: Settings, system_prompt: str) -> "Generator":
        """Build a generator from application settings and a system prompt."""
        return cls(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.chat_model,
            system_prompt=system_prompt,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
        )

    async def generate(self, question: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        """Generate a grounded answer for the question from the given chunks."""
        context = build_context(chunks)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": build_user_prompt(question, context)},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        content = response.choices[0].message.content or ""
        total_tokens = _total_tokens(response)
        logger.info("generated answer of %d chars, %d tokens", len(content), total_tokens)
        return GeneratedAnswer(text=content, total_tokens=total_tokens)
