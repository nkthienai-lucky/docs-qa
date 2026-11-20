from __future__ import annotations

import json
import math
from typing import Protocol
from urllib.request import Request, urlopen


class Embedder(Protocol):
    """Anything that can turn text into fixed-size unit vectors."""

    model: str
    backend: str
    dim: int
    query_instruction: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, q: str) -> list[float]: ...


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return [x / norm for x in vector]


class EmbeddingError(Exception):
    """Sanitized API embedding failure."""


class OpenAICompatibleEmbedder:
    backend = "openai_compatible"

    def __init__(
        self,
        host: str,
        model: str,
        token: str,
        query_instruction: str,
        timeout: float = 30.0,
    ) -> None:
        from . import provider_http

        self.endpoint = provider_http.provider_endpoint(host, "embeddings")
        self.model = model
        self.token = token
        self.query_instruction = query_instruction
        self.timeout = timeout
        self.dim = 0

    def embed_query(self, q: str) -> list[float]:
        text = (
            f"Instruct: {self.query_instruction}\nQuery: {q}"
            if self.query_instruction
            else q
        )
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from . import provider_http

        if not texts:
            return []
        result = None
        try:
            body = provider_http.post_json(
                self.endpoint,
                self.token,
                {"model": self.model, "input": texts},
                self.timeout,
            )
            data = body["data"]
            if not isinstance(data, list) or len(data) != len(texts):
                raise ValueError
            vectors = {}
            dimension = self.dim
            for item in data:
                index = item["index"]
                vector = item["embedding"]
                if (
                    type(index) is not int
                    or not 0 <= index < len(texts)
                    or index in vectors
                    or not isinstance(vector, list)
                    or not vector
                ):
                    raise ValueError
                if any(
                    type(x) not in (int, float) or not math.isfinite(x) for x in vector
                ):
                    raise ValueError
                if dimension and len(vector) != dimension:
                    raise ValueError
                dimension = len(vector)
                scale = max(abs(x) for x in vector)
                if scale == 0:
                    raise ValueError
                scaled = [x / scale for x in vector]
                norm = math.sqrt(sum(x * x for x in scaled))
                vectors[index] = [x / norm for x in scaled]
            result = [vectors[i] for i in range(len(texts))]
        except Exception:  # noqa: BLE001, S110 — upstream errors may contain secrets
            pass
        if result is None:
            raise EmbeddingError("embedding provider request failed")
        self.dim = dimension
        return result


class OllamaEmbedder:
    """Embedder backed by Ollama's `/api/embed` endpoint.

    Query texts are prefixed with the instruction template required by
    instruction-tuned embedding models (e.g. qwen3-embedding); document
    texts are embedded as-is. All vectors are L2-normalized client-side
    so cosine similarity reduces to a dot product.
    """

    backend = "ollama"

    def __init__(
        self, host: str, model: str, query_instruction: str, timeout: float = 60.0
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.query_instruction = query_instruction
        self.timeout = timeout
        self.dim = 0  # discovered from the first response

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, q: str) -> list[float]:
        text = f"Instruct: {self.query_instruction}\nQuery: {q}"
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        request = Request(
            f"{self.host}/api/embed",
            data=json.dumps({"model": self.model, "input": texts}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        vectors = body["embeddings"]
        self.dim = len(vectors[0])
        return [l2_normalize(vector) for vector in vectors]
