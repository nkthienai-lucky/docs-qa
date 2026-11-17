from __future__ import annotations

import logging
import math
from pathlib import Path

import pytest

from docsqa.eval.judges import (
    JudgeConfig,
    Sample,
    SampleScores,
    agreement_with_labels,
    score_samples,
)
from docsqa.settings import Settings


def test_nan_counted_not_dropped(monkeypatch, tmp_path: Path) -> None:
    def fake_evaluate(samples, cfg, settings):
        return [
            {"faithfulness": float("nan"), "answer_relevancy": 0.5, "context_precision": 1.0, "context_recall": 0.25}
        ]

    monkeypatch.setattr("docsqa.eval.judges._score_ragas", fake_evaluate)
    samples = [Sample("q", "a", ["context"], "truth")]

    scores = score_samples(samples, JudgeConfig("test", "local", "fake"), Settings())

    assert len(scores) == 1
    assert scores[0].faithfulness is None
    assert scores[0].answer_relevancy == 0.5


def test_judge_exception_message_contains_no_token_or_base_url(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("request failed for https://private.test with token secret-token")

    monkeypatch.setattr("docsqa.eval.judges._score_ragas", fail)
    with pytest.raises(RuntimeError) as error:
        score_samples([Sample("q", "a", ["c"], "g")], JudgeConfig("api", "api", "m"), Settings())

    assert "secret-token" not in str(error.value)
    assert "private.test" not in str(error.value)


def test_api_judge_requires_token_and_fails_clearly_without_it() -> None:
    with pytest.raises(ValueError, match="token"):
        from docsqa.eval.judges import make_ragas_llm
        make_ragas_llm(
            JudgeConfig("api", "api", "judge-model"),
            Settings(api_judge_base_url="https://gateway.example/v1", api_judge_token=None),
        )


def test_http_client_loggers_not_below_info() -> None:
    from docsqa.eval.judges import _raise_http_loggers

    _raise_http_loggers()
    for name in ("httpx", "httpcore", "openai"):
        assert logging.getLogger(name).level >= logging.INFO


def test_cache_directory_is_gitignored() -> None:
    gitignore = Path(__file__).parents[2] / ".gitignore"
    assert "eval/cache/" in gitignore.read_text(encoding="utf-8")


def test_score_samples_caches_results_without_credentials(monkeypatch, tmp_path: Path) -> None:
    calls = 0

    def fake_evaluate(samples, cfg, settings):
        nonlocal calls
        calls += 1
        return [{"faithfulness": 0.8}]

    monkeypatch.setattr("docsqa.eval.judges._score_ragas", fake_evaluate)
    monkeypatch.setattr("docsqa.eval.judges._JUDGE_CACHE_DIR", tmp_path)
    samples = [Sample("q", "a", ["context"], "truth")]
    settings = Settings(api_judge_base_url="https://private.test/v1", api_judge_token="secret-token")

    first = score_samples(samples, JudgeConfig("api", "api", "judge-model"), settings)
    second = score_samples(samples, JudgeConfig("api", "api", "judge-model"), settings)

    assert first == second
    assert calls == 1
    cache_contents = "".join(path.read_text(encoding="utf-8") for path in tmp_path.glob("*.json"))
    assert "private.test" not in cache_contents
    assert "secret-token" not in cache_contents


def test_ragas_embeddings_use_the_project_ollama_embedder(monkeypatch) -> None:
    captured = {}

    class FakeEmbedder:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def embed_documents(self, texts):
            return []

        def embed_query(self, text):
            return []

    class FakeWrapper:
        def __init__(self, embeddings):
            self.embeddings = embeddings

    monkeypatch.setattr("docsqa.eval.judges.OllamaEmbedder", FakeEmbedder)
    monkeypatch.setattr("ragas.embeddings.LangchainEmbeddingsWrapper", FakeWrapper)

    from docsqa.eval.judges import make_ragas_embeddings

    wrapper = make_ragas_embeddings(Settings(ollama_host="http://ollama.local", embed_model="qwen-test"))

    assert isinstance(wrapper.embeddings, object)
    assert captured == {
        "host": "http://ollama.local",
        "model": "qwen-test",
        "query_instruction": "Given a web search query, retrieve relevant passages",
    }


def test_agreement_with_labels_uses_faithfulness_threshold() -> None:
    from docsqa.eval.judges import HumanLabel

    scores = [
        SampleScores(0.9, 0.7, 0.5, 0.6),
        SampleScores(0.2, 0.7, 0.5, 0.6),
    ]
    labels = [HumanLabel("a", True), HumanLabel("b", False)]
    assert agreement_with_labels(scores, labels) == 1.0
