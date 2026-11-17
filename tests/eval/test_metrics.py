from __future__ import annotations

from docsqa.chunking import Chunk
from docsqa.eval.metrics import retrieval_metrics


def chunk(chunk_id: str, page_start: int, page_end: int, text: str) -> Chunk:
    return Chunk(chunk_id, "doc", text, page_start, page_end, None, int(chunk_id))


def test_recall_precision_mrr_on_hand_built_example() -> None:
    retrieved = [
        chunk("0", 1, 1, "not relevant"),
        chunk("1", 3, 4, "contains exact evidence"),
        chunk("2", 8, 8, "other"),
        chunk("3", 9, 9, "another evidence span"),
    ]

    scores = retrieval_metrics(
        retrieved, gold_pages=[4, 9], gold_spans=[], k=3
    )

    assert scores.recall_at_k == 1 / 2
    assert scores.precision_at_k == 1 / 3
    assert scores.reciprocal_rank == 0.5
