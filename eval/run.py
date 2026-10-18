"""Evaluation runner: ``python -m eval.run``.

Computes Hit@k, answer accuracy (via an LLM judge), refusal accuracy and
average latency over ``eval/questions.yaml``, and writes the results to
``eval/results/<date>.json``.
"""

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db.session import async_session_factory, engine
from app.ingest.embeddings import EmbeddingClient
from app.runtime import configure_event_loop
from app.services.client import build_openai_client
from app.services.generation import Generator, load_system_prompt
from app.services.retrieval import REFUSAL_ANSWER, best_similarity, search_similar

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVAL_DIR / "questions.yaml"
RESULTS_DIR = EVAL_DIR / "results"
JUDGE_PROMPT_FILENAME = "judge.md"


class Question(BaseModel):
    id: int
    question: str
    expected_file: str | None = None
    facts: list[str] = Field(default_factory=list)

    @property
    def in_scope(self) -> bool:
        return self.expected_file is not None


@dataclass
class QuestionResult:
    id: int
    question: str
    expected_file: str | None
    in_scope: bool
    hit: bool | None
    refused: bool
    answer_correct: bool | None
    latency_ms: float
    answer: str


@dataclass
class Metrics:
    hit_at_k: float
    answer_accuracy: float
    refusal_accuracy: float
    avg_latency_ms: float
    num_in_scope: int
    num_out_of_scope: int


class Judge:
    """LLM judge that scores an answer against the expected key facts."""

    def __init__(self, settings: Settings, prompt: str) -> None:
        self._client = build_openai_client(
            settings.llm_base_url, settings.llm_api_key, settings.llm_timeout_seconds
        )
        self._model = settings.chat_model
        self._prompt = prompt

    async def judge(self, question: str, facts: list[str], answer: str) -> bool:
        fact_lines = "\n".join(f"- {fact}" for fact in facts)
        user = f"Question: {question}\nKey facts:\n{fact_lines}\n\nAnswer:\n{answer}"
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
        )
        verdict = (response.choices[0].message.content or "").strip().upper()
        return verdict.startswith("CORRECT")


def load_questions(path: Path) -> list[Question]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [Question(**item) for item in raw]


async def evaluate(
    questions: list[Question],
    embedder: EmbeddingClient,
    generator: Generator,
    judge: Judge,
    session: AsyncSession,
    top_k: int,
    threshold: float,
) -> list[QuestionResult]:
    results: list[QuestionResult] = []
    for question in questions:
        started = time.perf_counter()
        query_embedding = (await embedder.embed([question.question]))[0]
        chunks = await search_similar(session, query_embedding, top_k)
        refused = best_similarity(chunks) < threshold

        hit: bool | None = None
        correct: bool | None = None
        answer = REFUSAL_ANSWER if refused else ""

        if question.in_scope:
            hit = question.expected_file in {chunk.file_path for chunk in chunks}
            if refused:
                correct = False
            else:
                generated = await generator.generate(question.question, chunks)
                answer = generated.text
                correct = await judge.judge(question.question, question.facts, answer)

        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        results.append(
            QuestionResult(
                id=question.id,
                question=question.question,
                expected_file=question.expected_file,
                in_scope=question.in_scope,
                hit=hit,
                refused=refused,
                answer_correct=correct,
                latency_ms=latency_ms,
                answer=answer,
            )
        )
    return results


def _ratio(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_metrics(results: list[QuestionResult]) -> Metrics:
    in_scope = [r for r in results if r.in_scope]
    out_of_scope = [r for r in results if not r.in_scope]
    return Metrics(
        hit_at_k=_ratio([r.hit for r in in_scope if r.hit is not None]),
        answer_accuracy=_ratio(
            [r.answer_correct for r in in_scope if r.answer_correct is not None]
        ),
        refusal_accuracy=_ratio([r.refused for r in out_of_scope]),
        avg_latency_ms=round(_mean([r.latency_ms for r in results]), 1),
        num_in_scope=len(in_scope),
        num_out_of_scope=len(out_of_scope),
    )


def save_results(results: list[QuestionResult], metrics: Metrics, settings: Settings) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    payload = {
        "date": date,
        "config": {
            "top_k": settings.top_k,
            "relevance_threshold": settings.relevance_threshold,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "chat_model": settings.chat_model,
        },
        "metrics": asdict(metrics),
        "questions": [asdict(result) for result in results],
    }
    output = RESULTS_DIR / f"{date}.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def print_summary(metrics: Metrics) -> None:
    print("=== Evaluation metrics ===")
    print(f"Hit@k:            {metrics.hit_at_k:.2%}")
    print(f"Answer accuracy:  {metrics.answer_accuracy:.2%}")
    print(f"Refusal accuracy: {metrics.refusal_accuracy:.2%}")
    print(f"Avg latency:      {metrics.avg_latency_ms:.0f} ms")
    print(f"Questions: {metrics.num_in_scope} in-scope, {metrics.num_out_of_scope} out-of-scope")


def main() -> None:
    configure_event_loop()
    settings = get_settings()
    questions = load_questions(QUESTIONS_FILE)
    system_prompt = load_system_prompt(settings.prompts_dir)
    judge_prompt = (
        (settings.prompts_dir / JUDGE_PROMPT_FILENAME).read_text(encoding="utf-8").strip()
    )

    embedder = EmbeddingClient.from_settings(settings)
    generator = Generator.from_settings(settings, system_prompt)
    judge = Judge(settings, judge_prompt)

    async def run() -> None:
        try:
            async with async_session_factory() as session:
                results = await evaluate(
                    questions,
                    embedder,
                    generator,
                    judge,
                    session,
                    settings.top_k,
                    settings.relevance_threshold,
                )
            metrics = compute_metrics(results)
            output = save_results(results, metrics, settings)
            print_summary(metrics)
            print(f"results saved to {output}")
        finally:
            await engine.dispose()

    asyncio.run(run())


if __name__ == "__main__":
    main()
