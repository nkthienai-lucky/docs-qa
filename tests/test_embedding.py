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


def _install_fake_urlopen(
    monkeypatch: pytest.MonkeyPatch, raw_vector: list[float]
) -> list[dict]:
    calls: list[dict] = []

    def fake_urlopen(request, timeout=None):
        body = json.loads(request.data.decode("utf-8"))
        calls.append({"url": request.full_url, "body": body})
        return _FakeResponse({"embeddings": [list(raw_vector) for _ in body["input"]]})

    monkeypatch.setattr(embedding_module, "urlopen", fake_urlopen)
    return calls


def test_query_gets_instruction_documents_do_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            (
                "Instruct: Given a web search query, retrieve relevant passages\n"
                "Query: cách cài đặt OCR"
            )
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


def api_embedder(monkeypatch, body, instruction="retrieve"):
    captured = []

    def post(*args):
        captured.append(args)
        return body

    monkeypatch.setattr("docsqa.provider_http.post_json", post)
    return embedding_module.OpenAICompatibleEmbedder(
        "https://host", "embed", "token", instruction
    ), captured


def test_api_embeddings_restore_order_and_preserve_inputs(monkeypatch):
    adapter, calls = api_embedder(
        monkeypatch,
        {
            "data": [
                {"index": 1, "embedding": [0, 4]},
                {"index": 0, "embedding": [3, 0]},
            ]
        },
    )
    assert adapter.embed_documents([" É  ", "B"]) == [[1, 0], [0, 1]]
    assert calls[0] == (
        "https://host/v1/embeddings",
        "token",
        {"model": "embed", "input": [" É  ", "B"]},
        30,
    )
    assert adapter.dim == 2
    assert adapter.backend == "openai_compatible"


@pytest.mark.parametrize(
    "instruction,expected", [("retrieve", "Instruct: retrieve\nQuery: Q"), ("", "Q")]
)
def test_api_embedding_query_instruction(monkeypatch, instruction, expected):
    adapter, calls = api_embedder(
        monkeypatch, {"data": [{"index": 0, "embedding": [3, 4]}]}, instruction
    )
    assert adapter.embed_documents([]) == []
    assert calls == []
    assert adapter.embed_query("Q") == pytest.approx([0.6, 0.8])
    assert calls[0][2]["input"] == [expected]


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        [{"index": True, "embedding": [1]}],
        [{"index": 1, "embedding": [1]}],
        [{"embedding": [1]}],
        [{"index": 0, "embedding": []}],
        [{"index": 0, "embedding": [0, 0]}],
        [{"index": 0, "embedding": [True]}],
        [{"index": 0, "embedding": ["1"]}],
        [{"index": 0, "embedding": [float("nan")]}],
        [{"index": 0, "embedding": [float("inf")]}],
    ],
)
def test_api_embedding_rejects_invalid_response_without_dimension_mutation(
    monkeypatch, data
):
    adapter, _ = api_embedder(monkeypatch, {"data": data})
    with pytest.raises(embedding_module.EmbeddingError) as caught:
        adapter.embed_documents(["a"])
    assert adapter.dim == 0
    assert str(caught.value) == "embedding provider request failed"
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "data",
    [
        [{"index": 0, "embedding": [1]}, {"index": 0, "embedding": [2]}],
        [{"index": 0, "embedding": [1]}, {"index": 1, "embedding": [1, 2]}],
        [{"index": 0, "embedding": [1]}],
    ],
)
def test_api_embedding_validates_entire_batch(monkeypatch, data):
    adapter, _ = api_embedder(monkeypatch, {"data": data})
    with pytest.raises(embedding_module.EmbeddingError):
        adapter.embed_documents(["a", "b"])
    assert adapter.dim == 0


@pytest.mark.parametrize("magnitude", [1e308, 1e-308])
def test_api_embedding_stable_normalization_and_dimension_lock(monkeypatch, magnitude):
    body = {"data": [{"index": 0, "embedding": [magnitude, magnitude]}]}
    adapter, _ = api_embedder(monkeypatch, body)
    assert adapter.embed_query("q") == pytest.approx([2**-0.5, 2**-0.5])
    body["data"][0]["embedding"] = [1]
    with pytest.raises(embedding_module.EmbeddingError):
        adapter.embed_query("q")
    assert adapter.dim == 2


def test_api_embedding_sanitizes_transport_error(monkeypatch):
    import traceback

    adapter, _ = api_embedder(monkeypatch, {})

    def fail(*args):
        raise RuntimeError("sentinel-token https://private sentinel-body")

    monkeypatch.setattr("docsqa.provider_http.post_json", fail)
    with pytest.raises(embedding_module.EmbeddingError) as caught:
        adapter.embed_query("q")
    assert caught.value.__context__ is None
    assert "sentinel" not in "".join(traceback.format_exception(caught.value))
