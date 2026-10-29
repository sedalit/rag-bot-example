"""Markdown chunking: split documents by headings, then cap size with overlap."""

import re
from dataclasses import dataclass

# A Markdown ATX heading: 1-6 leading '#' characters followed by a space.
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class Chunk:
    """A single retrievable piece of a document."""

    section: str
    content: str
    index: int


@dataclass(frozen=True)
class Section:
    """A heading path and the body text belonging to it."""

    title: str
    content: str


def split_sections(markdown: str) -> list[Section]:
    """Split markdown into sections grouped by their nearest heading path.

    Nested headings produce joined titles such as ``"Installation > Windows"``.
    Text before the first heading is kept in a section with an empty title.
    """
    sections: list[Section] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_title = ""

    def flush() -> None:
        nonlocal current_lines
        body = "\n".join(current_lines).strip()
        if body:
            sections.append(Section(title=current_title, content=body))
        current_lines = []

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        match = HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_title = " > ".join(t for _, t in heading_stack)
        else:
            current_lines.append(line)

    flush()
    return sections


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split one piece of text into overlapping character windows of ``chunk_size``."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - chunk_overlap
    return chunks


class MarkdownChunker:
    """Chunk markdown by headings, then by size, producing ordered chunks."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, markdown: str) -> list[Chunk]:
        """Return chunks with section metadata and a sequential per-document index."""
        chunks: list[Chunk] = []
        index = 0
        for section in split_sections(markdown):
            for piece in split_text(section.content, self.chunk_size, self.chunk_overlap):
                chunks.append(Chunk(section=section.title, content=piece, index=index))
                index += 1
        return chunks
