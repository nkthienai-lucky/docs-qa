from __future__ import annotations

import json
from pathlib import Path

import pytest

from docsqa.chunking import Chunk
from docsqa.eval.dataset import Question
from docsqa.eval.profile import profile_config, write_profile_csv
from docsqa.generation import OllamaGenerator
from docsqa.pipeline import AnswerResponse, RetrievalInfo
from docsqa.store import Hit


class FakeRetriever:
    def retrieve(self, query, doc_ids=None):
        return [Hit(Chunk("doc:0", "doc", "Evidence", 1, 1, None, 0), 0.9)]


class FakePipeline:
    final_k = 1

    def __init__(self):
        self.retrievers = {"v1": FakeRetriever()}
        self.calls = 0

    def ask(self, question, config, hits=None):
        self.calls += 1
        return AnswerResponse(
            "Answer [1]", False, [], RetrievalInfo(config, ["doc:0"], [0.9])
        )


def make_question(split="dev"):
    return Question(
        "q1", "Question?", "Answer", "fact", [1], ["evidence"], "2026-11-18", split
    )


def test_profile_is_dev_only_and_writes_timing_data(tmp_path: Path):
    pipeline = FakePipeline()
    out = tmp_path / "profile.json"
    samples = profile_config("v1", [make_question()], pipeline, out_path=out)
    assert len(samples) == 1
    assert samples[0].retrieval_s >= 0
    assert samples[0].generation_s >= 0
    assert json.loads(out.read_text(encoding="utf-8"))["kind"] == "dev_profiling_only"
    write_profile_csv(samples, tmp_path / "profile.csv")
    assert "retrieval_s" in (tmp_path / "profile.csv").read_text(encoding="utf-8")
    assert pipeline.calls == 1


def test_profile_rejects_held_out_questions(tmp_path: Path):
    with pytest.raises(ValueError, match="dev"):
        profile_config(
            "v1",
            [make_question("held_out")],
            FakePipeline(),
            out_path=tmp_path / "profile.json",
        )


def test_generator_bounds_output_and_rejects_invalid_limits():
    generator = OllamaGenerator("http://127.0.0.1:11434", "qwen3.5:9b", 4096)
    assert generator.num_predict == 256
    with pytest.raises(ValueError, match="num_ctx and num_predict"):
        OllamaGenerator("http://127.0.0.1:11434", "qwen3.5:9b", 4096, num_predict=0)


def test_generator_can_use_a_larger_output_budget():
    generator = OllamaGenerator(
        "http://127.0.0.1:11434", "qwen3.5:9b", 4096, num_predict=512
    )
    assert generator.num_predict == 512


__all__ = ["test_profile_is_dev_only_and_writes_timing_data"]
