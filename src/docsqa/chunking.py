from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .corpus import normalize_text


@dataclass(frozen=True)
class Block:
    text: str
    page: int
    heading: str | None = None


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    page_start: int
    page_end: int
    section: str | None
    ordinal: int


TokenCounter = Callable[[str], int]


def _sections(blocks: list[Block]) -> list[tuple[str | None, list[Block]]]:
    sections: list[tuple[str | None, list[Block]]] = []
    current_heading: str | None = None
    current: list[Block] = []
    for block in blocks:
        heading = normalize_text(block.heading) if block.heading else None
        if heading is not None and current:
            sections.append((current_heading, current))
            current = []
        if heading is not None:
            current_heading = heading
        if normalize_text(block.text):
            current.append(Block(normalize_text(block.text), block.page, current_heading))
    if current:
        sections.append((current_heading, current))
    return sections


def chunk_blocks(
    blocks: list[Block],
    doc_id: str,
    size: int,
    overlap: int,
    count_tokens: TokenCounter,
) -> list[Chunk]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("size must be positive and overlap must be smaller than size")

    chunks: list[Chunk] = []
    for section, section_blocks in _sections(blocks):
        token_pages: list[tuple[str, int]] = []
        for block in section_blocks:
            token_pages.extend((token, block.page) for token in block.text.split())
        if not token_pages:
            continue

        # The public token counter may be a model tokenizer. The built-in
        # window uses whitespace tokens, and validates the resulting text with
        # the supplied counter before accepting it.
        size - overlap
        start = 0
        while start < len(token_pages):
            end = min(len(token_pages), start + size)
            window = token_pages[start:end]
            text = " ".join(token for token, _ in window)
            while count_tokens(text) > size and end > start + 1:
                end -= 1
                window = token_pages[start:end]
                text = " ".join(token for token, _ in window)
            ordinal = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:{ordinal}",
                    doc_id=doc_id,
                    text=text,
                    page_start=min(page for _, page in window),
                    page_end=max(page for _, page in window),
                    section=section,
                    ordinal=ordinal,
                )
            )
            if end == len(token_pages):
                break
            start = max(start + 1, end - overlap)
    return chunks
