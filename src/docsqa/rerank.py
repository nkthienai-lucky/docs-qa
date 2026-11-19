from __future__ import annotations

from sentence_transformers import CrossEncoder

from .store import Hit

_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
_DEFAULT_DEVICE = "cpu"
_CUDA_BATCH_SIZE = 2
_CPU_BATCH_SIZE = 32


def _validated_batch_size(batch_size: int) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return batch_size


def _resolve_device(device: str) -> str:
    if device == "auto":
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda", "cuda:0"}:
        raise ValueError("device must be auto, cpu, cuda, or cuda:0")
    if device.startswith("cuda"):
        try:
            import torch
        except ImportError as error:
            raise ValueError("CUDA requested but PyTorch is unavailable") from error
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but no CUDA device is available")
    return device


class Reranker:
    """Cross-encoder wrapper that reranks hits by model score."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        device: str = _DEFAULT_DEVICE,
        batch_size: int | None = None,
    ) -> None:
        self.model_name = model
        self.device = _resolve_device(device)
        default_batch_size = (
            _CUDA_BATCH_SIZE if self.device.startswith("cuda") else _CPU_BATCH_SIZE
        )
        self.batch_size = _validated_batch_size(
            batch_size if batch_size is not None else default_batch_size
        )
        self.model: CrossEncoder = CrossEncoder(model, device=self.device)

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if top_k <= 0 or not hits:
            return []
        pairs = [(query, hit.chunk.text) for hit in hits]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        order = sorted(range(len(hits)), key=lambda i: (-float(scores[i]), i))
        selected = order[:top_k]
        return [Hit(chunk=hits[i].chunk, score=float(scores[i])) for i in selected]


__all__ = ["Reranker"]
