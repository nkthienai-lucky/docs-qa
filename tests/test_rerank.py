from __future__ import annotations

import numpy
import pytest

from docsqa.chunking import Chunk
from docsqa.rerank import Reranker
from docsqa.store import Hit


def _hit(ordinal: int, text: str, score: float = 0.0, doc_id: str = "doc") -> Hit:
    return Hit(
        chunk=Chunk(
            chunk_id=f"{doc_id}:{ordinal}",
            doc_id=doc_id,
            text=text,
            page_start=1,
            page_end=2,
            section=None,
            ordinal=ordinal,
        ),
        score=score,
    )


class FakeCrossEncoder:
    """CrossEncoder stand-in the test controls explicitly."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs, **kwargs):
        self.pairs.extend(pairs)
        return numpy.array([self._scores[text] for _, text in pairs])


def test_reranker_reorders_by_cross_encoder_score() -> None:
    # Vector order (by retrieval score) puts the filler chunk first; the
    # cross-encoder scores put the relevant chunk first. Reranking must
    # reorder by cross-encoder score, keep non-negative scores, and attach
    # the score to the returned Hits.
    fake = FakeCrossEncoder(
        {
            "filler about classrooms and pencils": 0.01,
            "lend means to give something temporarily": 0.95,
            "a third unrelated passage about weather": 0.30,
        }
    )
    reranker = Reranker.__new__(Reranker)  # skip model load
    reranker.model = fake
    reranker.device = "cpu"

    hits = [
        _hit(0, "filler about classrooms and pencils", score=0.61),
        _hit(1, "lend means to give something temporarily", score=0.50),
        _hit(2, "a third unrelated passage about weather", score=0.42),
    ]

    reranked = reranker.rerank("what does lend mean?", hits, top_k=2)

    assert [hit.chunk.chunk_id for hit in reranked] == ["doc:1", "doc:2"]
    assert [hit.score for hit in reranked] == [0.95, 0.30]
    # The model saw (query, passage) pairs for every candidate.
    assert fake.pairs == [
        ("what does lend mean?", "filler about classrooms and pencils"),
        ("what does lend mean?", "lend means to give something temporarily"),
        ("what does lend mean?", "a third unrelated passage about weather"),
    ]


def test_reranker_runs_on_cpu_by_default() -> None:
    # The device must default to CPU and be passed through to CrossEncoder.
    # We patch CrossEncoder in docsqa.rerank to capture the kwargs without
    # loading weights.
    import docsqa.rerank as rerank_module

    captured: dict[str, object] = {}

    class RecordingCrossEncoder:
        def __init__(self, model, **kwargs) -> None:
            captured["model"] = model
            captured.update(kwargs)

    original = rerank_module.CrossEncoder
    rerank_module.CrossEncoder = RecordingCrossEncoder
    try:
        reranker = Reranker()
    finally:
        rerank_module.CrossEncoder = original

    assert reranker.device == "cpu"
    assert captured["model"] == "BAAI/bge-reranker-v2-m3"
    assert captured["device"] == "cpu"


def test_rerank_preserves_hit_chunks_and_truncates_to_top_k() -> None:
    fake = FakeCrossEncoder({"a": 0.5, "b": 0.9, "c": 0.1})
    reranker = Reranker.__new__(Reranker)
    reranker.model = fake
    reranker.device = "cpu"
    original = [
        _hit(0, "a", score=0.9),
        _hit(1, "b", score=0.8),
        _hit(2, "c", score=0.7),
    ]

    reranked = reranker.rerank("q", original, top_k=1)

    assert len(reranked) == 1
    assert reranked[0].chunk.chunk_id == "doc:1"
    assert reranked[0].chunk is original[1].chunk
