from __future__ import annotations

from dataclasses import dataclass

from ..chunking import Chunk


@dataclass(frozen=True)
class RetrievalScores:
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float


def _is_relevant(chunk: Chunk, gold_pages: set[int], gold_spans: list[str]) -> bool:
    overlaps_page = any(
        chunk.page_start <= page <= chunk.page_end for page in gold_pages
    )
    contains_span = any(span.casefold() in chunk.text.casefold() for span in gold_spans if span)
    return overlaps_page or contains_span


def retrieval_metrics(
    retrieved: list[Chunk],
    gold_pages: list[int],
    gold_spans: list[str],
    k: int,
) -> RetrievalScores:
    if k <= 0:
        raise ValueError("k must be positive")
    top = retrieved[:k]
    gold = set(gold_pages) | {f"span:{span.casefold()}" for span in gold_spans if span}
    if not gold:
        raise ValueError("at least one gold page or span is required")

    relevant = [_is_relevant(chunk, set(gold_pages), gold_spans) for chunk in top]
    covered_pages = {
        page
        for page in gold_pages
        if any(chunk.page_start <= page <= chunk.page_end for chunk in top)
    }
    covered_spans = {
        span
        for span in gold_spans
        if any(span.casefold() in chunk.text.casefold() for chunk in top)
    }
    recall = len(covered_pages) + len(covered_spans)
    precision = sum(relevant) / k
    first_relevant = next((index for index, value in enumerate(relevant, 1) if value), None)
    return RetrievalScores(
        recall_at_k=recall / len(gold),
        precision_at_k=precision,
        reciprocal_rank=1 / first_relevant if first_relevant is not None else 0.0,
    )
