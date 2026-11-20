"""Cohere-style full-list API reranking without local model imports."""

from __future__ import annotations

import math

from . import provider_http
from .store import Hit


class RerankerError(Exception):
    """Sanitized API reranking failure."""


class ApiReranker:
    def __init__(
        self, host: str, model: str, token: str, timeout: float = 30.0
    ) -> None:
        self.endpoint = provider_http.provider_endpoint(host, "rerank")
        self.model = model
        self.token = token
        self.timeout = timeout

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits or top_k <= 0:
            return []
        result = None
        try:
            body = provider_http.post_json(
                self.endpoint,
                self.token,
                {
                    "model": self.model,
                    "query": query,
                    "documents": [h.chunk.text for h in hits],
                    "top_n": len(hits),
                },
                self.timeout,
            )
            rows = body["results"]
            if not isinstance(rows, list) or len(rows) != len(hits):
                raise ValueError
            scores = {}
            for row in rows:
                index, score = row["index"], row["relevance_score"]
                if (
                    type(index) is not int
                    or not 0 <= index < len(hits)
                    or index in scores
                    or type(score) not in (int, float)
                    or not math.isfinite(score)
                ):
                    raise ValueError
                scores[index] = float(score)
            order = sorted(scores, key=lambda i: (-scores[i], i))[:top_k]
            result = [Hit(chunk=hits[i].chunk, score=scores[i]) for i in order]
        except Exception:  # noqa: BLE001, S110 — upstream errors may contain secrets
            pass
        if result is None:
            raise RerankerError("reranking provider request failed")
        return result
