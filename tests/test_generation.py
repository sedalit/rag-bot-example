"""Unit tests for prompt assembly."""

from app.config import BASE_DIR
from app.services.generation import build_context, build_user_prompt, load_system_prompt
from app.services.retrieval import RetrievedChunk


def _chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            content="first content", section="A", file_path="a.md", chunk_index=0, similarity=0.9
        ),
        RetrievedChunk(
            content="second content", section="B", file_path="b.md", chunk_index=1, similarity=0.8
        ),
    ]


def test_build_context_numbers_blocks_with_metadata():
    context = build_context(_chunks())
    assert "[1] a.md — A\nfirst content" in context
    assert "[2] b.md — B\nsecond content" in context
    assert context.index("[1]") < context.index("[2]")


def test_build_user_prompt_wraps_context_and_question():
    prompt = build_user_prompt("How do I start?", "CONTEXT")
    assert prompt == "Context:\nCONTEXT\n\nQuestion: How do I start?"


def test_load_system_prompt_reads_committed_file():
    prompt = load_system_prompt(BASE_DIR / "app" / "prompts")
    assert prompt
    assert "context" in prompt.lower()
    assert "[1]" in prompt or "cite" in prompt.lower()
