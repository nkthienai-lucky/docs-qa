from __future__ import annotations

import json
import math
from typing import Self

import pytest

import docsqa.embedding as embedding_module
from docsqa.embedding import OllamaEmbedder


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def _install_fake_urlopen(monkeypatch: pytest.MonkeyPatch, raw_vector: list[float]) -> list[dict]:
    calls: list[dict] = []

    def fake_urlopen(request, timeout=None):
        body = json.loads(request.data.decode("utf-8"))
        calls.append({"url": request.full_url, "body": body})
        return _FakeResponse({"embeddings": [list(raw_vector) for _ in body["input"]]})

    monkeypatch.setattr(embedding_module, "urlopen", fake_urlopen)
    return calls


def test_query_gets_instruction_documents_do_not(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_urlopen(monkeypatch, [3.0, 4.0, 0.0, 0.0])
    embedder = OllamaEmbedder(
        host="http://127.0.0.1:11434",
        model="qwen3-embedding:0.6b",
        query_instruction="Given a web search query, retrieve relevant passages",
    )

    embedder.embed_query("cách cài đặt OCR")
    embedder.embed_documents(["văn bản thứ nhất", "văn bản thứ hai"])

    query_call, documents_call = calls
    assert query_call["url"] == "http://127.0.0.1:11434/api/embed"
    assert query_call["body"] == {
        "model": "qwen3-embedding:0.6b",
        "input": [
            ("Instruct: Given a web search query, retrieve relevant passages\n"
            "Query: cách cài đặt OCR")
        ],
    }
    assert documents_call["body"]["input"] == ["văn bản thứ nhất", "văn bản thứ hai"]
    assert not any("Instruct:" in text for text in documents_call["body"]["input"])


def test_vectors_are_unit_norm(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_urlopen(monkeypatch, [0.0, 3.0, 4.0, 0.0])
    embedder = OllamaEmbedder(
        host="http://127.0.0.1:11434",
        model="qwen3-embedding:0.6b",
        query_instruction="retrieve relevant passages",
    )

    documents = embedder.embed_documents(["alpha", "beta"])
    query = embedder.embed_query("gamma")

    for vector in [*documents, query]:
        assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0)
    assert query[1] == pytest.approx(0.6)
    assert query[2] == pytest.approx(0.8)
    assert embedder.dim == 4
    assert embedder.backend == "ollama"
    assert embedder.model == "qwen3-embedding:0.6b"
