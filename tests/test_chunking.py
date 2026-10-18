"""Unit tests for Markdown chunking."""

import pytest

from app.ingest.chunking import MarkdownChunker, split_sections, split_text


def test_split_sections_single_heading():
    sections = split_sections("# Title\n\nbody text\n")
    assert len(sections) == 1
    assert sections[0].title == "Title"
    assert sections[0].content == "body text"


def test_split_sections_nested_headings_join_path():
    markdown = "# A\nintro\n## B\ndetail\n### C\nleaf\n"
    sections = split_sections(markdown)
    assert [s.title for s in sections] == ["A", "A > B", "A > B > C"]


def test_split_sections_sibling_headings_reset_level():
    markdown = "# One\nfirst\n# Two\nsecond\n"
    sections = split_sections(markdown)
    assert [s.title for s in sections] == ["One", "Two"]


def test_split_sections_preamble_gets_empty_title():
    sections = split_sections("before any heading\n# H\nbody\n")
    assert sections[0].title == ""
    assert sections[0].content == "before any heading"
    assert sections[1].title == "H"


def test_split_sections_ignores_empty_sections():
    sections = split_sections("# Only heading\n")
    assert sections == []


def test_split_sections_heading_text_is_stripped():
    sections = split_sections("##   Spaced title   \nbody\n")
    assert sections[0].title == "Spaced title"


def test_split_text_short_returns_single():
    assert split_text("hello", 800, 100) == ["hello"]


def test_split_text_empty_and_whitespace():
    assert split_text("", 800, 100) == []
    assert split_text("   \n  ", 800, 100) == []


def test_split_text_exact_size_is_single_chunk():
    text = "a" * 800
    assert split_text(text, 800, 100) == [text]


def test_split_text_overlaps_chunks():
    text = "0123456789"
    chunks = split_text(text, 6, 2)
    assert chunks == ["012345", "456789"]


def test_split_text_respects_max_size():
    chunks = split_text("x" * 1000, 800, 100)
    assert all(len(c) <= 800 for c in chunks)
    assert len(chunks) == 2


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (10, 10), (10, 11), (10, -1)])
def test_split_text_invalid_parameters_raise(size: int, overlap: int):
    with pytest.raises(ValueError):
        split_text("text", size, overlap)


def test_chunker_produces_section_metadata_and_order():
    chunker = MarkdownChunker(chunk_size=800, chunk_overlap=100)
    chunks = chunker.chunk("# First\nalpha\n\n# Second\nbeta\n")
    assert [(c.section, c.index) for c in chunks] == [("First", 0), ("Second", 1)]


def test_chunker_splits_long_sections():
    chunker = MarkdownChunker(chunk_size=6, chunk_overlap=2)
    chunks = chunker.chunk("# S\n0123456789\n")
    assert [c.content for c in chunks] == ["012345", "456789"]
    assert all(c.section == "S" for c in chunks)
    assert [c.index for c in chunks] == [0, 1]


def test_chunker_invalid_parameters_raise():
    with pytest.raises(ValueError):
        MarkdownChunker(chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError):
        MarkdownChunker(chunk_size=100, chunk_overlap=100)
