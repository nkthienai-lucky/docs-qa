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


class OllamaEmbedder:
    """Embedder backed by Ollama's `/api/embed` endpoint.

    Query texts are prefixed with the instruction template required by
    instruction-tuned embedding models (e.g. qwen3-embedding); document
    texts are embedded as-is. All vectors are L2-normalized client-side
    so cosine similarity reduces to a dot product.
    """

    backend = "ollama"

    def __init__(self, host: str, model: str, query_instruction: str, timeout: float = 60.0) -> None:
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
