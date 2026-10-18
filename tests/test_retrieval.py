"""Unit tests for retrieval threshold logic."""

from app.services.retrieval import RetrievedChunk, best_similarity, meets_threshold


def _chunk(similarity: float) -> RetrievedChunk:
    return RetrievedChunk(
        content="c", section="s", file_path="f.md", chunk_index=0, similarity=similarity
    )


def test_best_similarity_empty_is_zero():
    assert best_similarity([]) == 0.0


def test_best_similarity_returns_top_chunk():
    chunks = [_chunk(0.8), _chunk(0.7)]
    assert best_similarity(chunks) == 0.8


def test_meets_threshold_empty_is_false():
    assert meets_threshold([], 0.5) is False


def test_meets_threshold_boundary_values():
    assert meets_threshold([_chunk(0.5)], 0.5) is True
    assert meets_threshold([_chunk(0.49)], 0.5) is False
    assert meets_threshold([_chunk(0.9)], 0.5) is True


def test_meets_threshold_uses_only_best_chunk():
    chunks = [_chunk(0.9), _chunk(0.1)]
    assert meets_threshold(chunks, 0.5) is True
