from __future__ import annotations

import numpy
import pytest

from docsqa.chunking import Chunk
from docsqa.rerank import Reranker, _resolve_device
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
        self.kwargs: dict[str, object] = {}

    def predict(self, pairs, **kwargs):
        self.pairs.extend(pairs)
        self.kwargs.update(kwargs)
        return numpy.array([self._scores[text] for _, text in pairs])


def test_reranker_reorders_by_cross_encoder_score() -> None:
    fake = FakeCrossEncoder(
        {
            "filler about classrooms and pencils": 0.01,
            "lend means to give something temporarily": 0.95,
            "a third unrelated passage about weather": 0.30,
        }
    )
    reranker = Reranker.__new__(Reranker)
    reranker.model = fake
    reranker.device = "cpu"
    reranker.batch_size = 32
    hits = [
        _hit(0, "filler about classrooms and pencils", score=0.61),
        _hit(1, "lend means to give something temporarily", score=0.50),
        _hit(2, "a third unrelated passage about weather", score=0.42),
    ]

    reranked = reranker.rerank("what does lend mean?", hits, top_k=2)

    assert [hit.chunk.chunk_id for hit in reranked] == ["doc:1", "doc:2"]
    assert [hit.score for hit in reranked] == [0.95, 0.30]
    assert fake.pairs == [
        ("what does lend mean?", "filler about classrooms and pencils"),
        ("what does lend mean?", "lend means to give something temporarily"),
        ("what does lend mean?", "a third unrelated passage about weather"),
    ]
    assert fake.kwargs["batch_size"] == 32


def test_reranker_device_batch_defaults_and_overrides(monkeypatch) -> None:
    import sentence_transformers
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    captured: dict[str, object] = {}

    class RecordingCrossEncoder:
        def __init__(self, model, **kwargs) -> None:
            captured["model"] = model
            captured.update(kwargs)

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", RecordingCrossEncoder)
    cpu_reranker = Reranker(device="cpu")
    cuda_reranker = Reranker(device="cuda")
    overridden = Reranker(device="cuda", batch_size=4)

    assert cpu_reranker.device == "cpu"
    assert cpu_reranker.batch_size == 32
    assert cuda_reranker.device == "cuda"
    assert cuda_reranker.batch_size == 2
    assert overridden.batch_size == 4
    assert captured["model"] == "BAAI/bge-reranker-v2-m3"
    assert captured["device"] == "cuda"


def test_reranker_auto_device_falls_back_to_cpu(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _resolve_device("auto") == "cpu"
    with pytest.raises(ValueError, match="CUDA requested"):
        _resolve_device("cuda")


def test_reranker_auto_device_selects_cuda(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _resolve_device("auto") == "cuda"


def test_reranker_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        Reranker(batch_size=0)


def test_rerank_preserves_hit_chunks_and_truncates_to_top_k() -> None:
    fake = FakeCrossEncoder({"a": 0.5, "b": 0.9, "c": 0.1})
    reranker = Reranker.__new__(Reranker)
    reranker.model = fake
    reranker.device = "cpu"
    reranker.batch_size = 32
    original = [
        _hit(0, "a", score=0.9),
        _hit(1, "b", score=0.8),
        _hit(2, "c", score=0.7),
    ]

    reranked = reranker.rerank("q", original, top_k=1)

    assert len(reranked) == 1
    assert reranked[0].chunk.chunk_id == "doc:1"
    assert reranked[0].chunk is original[1].chunk
