import subprocess
import sys
import traceback

import pytest

from docsqa.chunking import Chunk
from docsqa.store import Hit


def hits():
    return [Hit(Chunk(str(i), "doc", str(i), 1, 1, None, i), 0) for i in range(3)]


def adapter(monkeypatch, body):
    from docsqa.api_rerank import ApiReranker

    calls = []

    def post(*args):
        calls.append(args)
        return body

    monkeypatch.setattr("docsqa.provider_http.post_json", post)
    return ApiReranker("https://host", "rank", "token"), calls


def test_full_list_request_order_ties_and_chunk_identity(monkeypatch):
    ranker, calls = adapter(
        monkeypatch,
        {
            "results": [
                {"index": 2, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": -0.5},
            ]
        },
    )
    original = hits()
    result = ranker.rerank("q", original, 20)
    assert [h.chunk for h in result] == [original[i].chunk for i in [1, 2, 0]]
    assert result[0].chunk is original[1].chunk
    assert [h.score for h in result] == [0.9, 0.9, -0.5]
    assert calls[0] == (
        "https://host/v1/rerank",
        "token",
        {"model": "rank", "query": "q", "documents": ["0", "1", "2"], "top_n": 3},
        30,
    )
    assert len(ranker.rerank("q", original, 1)) == 1


def test_empty_rerank_no_http(monkeypatch):
    ranker, calls = adapter(monkeypatch, {})
    assert ranker.rerank("q", [], 3) == []
    assert ranker.rerank("q", hits(), 0) == []
    assert ranker.rerank("q", hits(), -1) == []
    assert calls == []


@pytest.mark.parametrize(
    "results",
    [
        None,
        [],
        [{"index": 0, "relevance_score": 1}],
        [{"index": 0, "relevance_score": 1}] * 3,
        [{"index": i, "relevance_score": 1} for i in [0, 1, 3]],
        [{"index": i, "relevance_score": 1} for i in [False, 1, 2]],
        [{"index": i, "relevance_score": True} for i in range(3)],
        [{"index": i, "relevance_score": float("nan")} for i in range(3)],
        [{"index": i, "relevance_score": float("inf")} for i in range(3)],
        [{"index": i} for i in range(3)],
    ],
)
def test_invalid_rerank_responses(monkeypatch, results):
    from docsqa.api_rerank import RerankerError

    ranker, _ = adapter(monkeypatch, {"results": results})
    with pytest.raises(RerankerError) as caught:
        ranker.rerank("q", hits(), 2)
    assert str(caught.value) == "reranking provider request failed"
    assert caught.value.__context__ is None


def test_rerank_transport_error_redacted(monkeypatch):
    from docsqa.api_rerank import RerankerError

    ranker, _ = adapter(monkeypatch, {})

    def fail(*args):
        raise TimeoutError("sentinel-token https://private sentinel-body")

    monkeypatch.setattr("docsqa.provider_http.post_json", fail)
    with pytest.raises(RerankerError) as caught:
        ranker.rerank("q", hits(), 2)
    assert caught.value.__context__ is None
    assert "sentinel" not in "".join(traceback.format_exception(caught.value))


def test_api_imports_do_not_load_local_ml():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import docsqa.api_rerank, docsqa.runtime, docsqa.rerank; import sys; "
                "assert 'sentence_transformers' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
