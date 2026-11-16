from __future__ import annotations

from sentence_transformers import CrossEncoder

from .store import Hit

# Reranker runs on CPU by design: the 8GB GPU is reserved for the generator
# and embedder. The model is loaded once at construction and reused for every
# query; ``predict`` returns the raw CrossEncoder scores, which for
# bge-reranker-v2-m3 are sigmoid probabilities (probe confirmed 0.92+ for
# relevant pairs, ~1e-5 for irrelevant ones).
_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
_DEFAULT_DEVICE = "cpu"


class Reranker:
    """Cross-encoder reranker wrapping ``sentence_transformers.CrossEncoder``.

    Construction loads the model (and downloads it on first run). Each
    ``rerank`` call scores ``(query, passage)`` pairs for every candidate Hit
    and returns the top ``top_k`` in descending score order, with the
    cross-encoder score attached to each returned Hit (the original Hit's
    retrieval score is replaced). The chunk objects are reused as-is.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        device: str = _DEFAULT_DEVICE,
    ) -> None:
        self.model_name = model
        self.device = device
        self.model: CrossEncoder = CrossEncoder(model, device=device)

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if top_k <= 0:
            return []
        if not hits:
            return []
        pairs = [(query, hit.chunk.text) for hit in hits]
        scores = self.model.predict(pairs)
        # Order candidate indices by descending cross-encoder score, stable
        # on original position so ties keep the retrieval order.
        order = sorted(
            range(len(hits)),
            key=lambda i: (-float(scores[i]), i),
        )
        selected = order[:top_k]
        return [
            Hit(chunk=hits[i].chunk, score=float(scores[i])) for i in selected
        ]
