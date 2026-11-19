from __future__ import annotations

import json

import pytest

from docsqa.chunking import Chunk
from docsqa.eval.dataset import Question
from docsqa.eval.judges import JudgeConfig, SampleScores
from docsqa.eval.runlog import RunConfig, RunLog
from docsqa.eval.runner import (
    load_sample_results,
    run_config,
    run_configs,
    run_held_out_once,
    save_sample_results,
    score_run,
)
from docsqa.pipeline import AnswerResponse, RetrievalInfo


class FakePipeline:
    final_k = 2

    def __init__(self):
        self.calls = []

    def ask(self, question, config, doc_ids=None, hits=None):
        self.calls.append((question, config, doc_ids, hits))
        return AnswerResponse(
            answer="Answer [1]",
            abstained=False,
            citations=[],
            retrieval=RetrievalInfo(config, ["doc:0", "doc:1"], [0.9, 0.8]),
        )


class FakeRetriever:
    def retrieve(self, question, doc_ids=None):
        return [
            type(
                "Hit",
                (),
                {
                    "chunk": Chunk("doc:0", "doc", "Evidence here", 1, 1, None, 0),
                    "score": 0.9,
                },
            )(),
            type(
                "Hit",
                (),
                {"chunk": Chunk("doc:1", "doc", "Other", 2, 2, None, 1), "score": 0.8},
            )(),
        ]


def question(split="dev", category="fact"):
    return Question(
        "q1",
        "Question?",
        "Ground truth",
        category,
        [] if category == "out_of_scope" else [1],
        [] if category == "out_of_scope" else ["Evidence"],
        "2026-11-17",
        split,
    )


def run_config_fixture():
    return RunConfig("v1", None, 512, 64, 2, 50, 60, "prompt", "dev", generation_k=2)


def test_all_configs_receive_same_final_k_generation_k_and_prompt_hash():
    from docsqa.eval.runner import validate_comparable_configs

    configs = [
        RunConfig(cfg, None, 512, 64, 5, 50, 60, "same-prompt", "dev")
        for cfg in ("v1", "v2a", "v2b")
    ]
    validate_comparable_configs(configs)

    with pytest.raises(ValueError, match="prompt_hash"):
        validate_comparable_configs(
            [configs[0], RunConfig("v2a", None, 512, 64, 5, 50, 60, "different", "dev")]
        )
    with pytest.raises(ValueError, match="generation_k"):
        validate_comparable_configs(
            [configs[0], RunConfig("v2a", None, 512, 64, 5, 50, 60, "same-prompt", "dev", generation_k=1)]
        )


def test_sample_results_round_trip_checkpoint(tmp_path):
    from docsqa.eval.metrics import RetrievalScores
    from docsqa.eval.runner import SampleResult

    result = SampleResult(
        question=question(),
        config="v1",
        answer="Grounded answer",
        abstained=False,
        retrieved_chunk_ids=["doc:0"],
        retrieval_scores=[0.9],
        retrieval=RetrievalScores(1.0, 0.2, 1.0),
        contexts=["Evidence here"],
    )
    path = tmp_path / "dev" / "v1.json"
    save_sample_results([result], path)
    assert load_sample_results(path) == [result]
    assert not path.with_suffix(".json.tmp").exists()


def test_runner_scores_only_contexts_seen_by_generator(tmp_path):
    fake_pipeline = FakePipeline()
    fake_pipeline.generation_k = 1
    pipeline = type(
        "Pipeline",
        (),
        {
            "final_k": 2,
            "generation_k": 1,
            "retrievers": {"v1": FakeRetriever()},
            "ask": lambda self, q, config, doc_ids=None, hits=None: fake_pipeline.ask(
                q, config, doc_ids, hits
            ),
        },
    )()
    cfg = RunConfig("v1", None, 512, 64, 2, 50, 60, "prompt", "dev", generation_k=1)
    result = run_config(
        "v1", [question()], pipeline, cfg, RunLog(tmp_path / "runs.jsonl")
    )
    assert result[0].contexts == ["Evidence here"]
    record = json.loads((tmp_path / "runs.jsonl").read_text())
    assert record["metadata"]["generation_k"] == 1
    assert record["config"]["generation_k"] == 1


def batch_config(name, split="dev", prompt_hash="prompt"):
    return RunConfig(name, None, 512, 64, 2, 50, 60, prompt_hash, split, generation_k=2)


def test_run_configs_checkpoints_dev_results_incrementally(tmp_path):
    fake_pipeline = FakePipeline()
    retriever = FakeRetriever()
    pipeline = type(
        "Pipeline",
        (),
        {
            "final_k": 2,
            "generation_k": 2,
            "retrievers": {"v1": retriever},
            "ask": lambda self, q, config, doc_ids=None, hits=None: fake_pipeline.ask(
                q, config, doc_ids, hits
            ),
        },
    )()
    cfg = batch_config("v1")
    results = run_configs(
        [question()], pipeline, [cfg], RunLog(tmp_path / "runs.jsonl"),
        checkpoint_dir=tmp_path / "checkpoints",
    )
    path = tmp_path / "checkpoints" / "v1-dev-round0.json"
    assert load_sample_results(path) == results["v1"]


def test_run_configs_validates_comparability_before_any_retrieval(tmp_path):
    retriever = FakeRetriever()
    pipeline = type(
        "Pipeline",
        (),
        {
            "final_k": 2,
            "retrievers": {"v1": retriever, "v2a": retriever},
            "ask": lambda self, q, config, doc_ids=None, hits=None: FakePipeline().ask(
                q, config, doc_ids, hits
            ),
        },
    )()
    calls = []
    original_retrieve = retriever.retrieve

    def capture_retrieve(*args, **kwargs):
        calls.append(args[0])
        return original_retrieve(*args, **kwargs)

    retriever.retrieve = capture_retrieve
    configs = [batch_config("v1"), batch_config("v2a", prompt_hash="other")]
    with pytest.raises(ValueError, match="prompt_hash"):
        run_configs([question()], pipeline, configs, RunLog(tmp_path / "runs.jsonl"))
    assert calls == []


def test_held_out_batch_blocks_reuse_before_retrieval(tmp_path):
    retriever = FakeRetriever()
    pipeline = type(
        "Pipeline",
        (),
        {
            "final_k": 2,
            "retrievers": {
                name: retriever for name in ("v1", "v2a", "v2b", "v2b_norerank")
            },
            "ask": lambda self, q, config, doc_ids=None, hits=None: FakePipeline().ask(
                q, config, doc_ids, hits
            ),
        },
    )()
    log = RunLog(tmp_path / "runs.jsonl", held_out_ids={"q1"})
    held_configs = [
        batch_config(name, "held_out") for name in ("v1", "v2a", "v2b", "v2b_norerank")
    ]
    run_held_out_once([question("held_out")], pipeline, held_configs, log)
    calls = []
    retriever.retrieve = lambda *args, **kwargs: calls.append(args[0]) or []
    with pytest.raises(ValueError, match="already recorded"):
        run_held_out_once([question("held_out")], pipeline, held_configs, log)
    assert calls == []


def test_held_out_batch_validation_failure_writes_no_partial_records(tmp_path):
    retriever = FakeRetriever()
    pipeline = type(
        "Pipeline",
        (),
        {
            "final_k": 2,
            "generation_k": 2,
            "retrievers": {
                name: retriever for name in ("v1", "v2a", "v2b", "v2b_norerank")
            },
            "ask": lambda self, q, config, doc_ids=None, hits=None: FakePipeline().ask(
                q, config, doc_ids, hits
            ),
        },
    )()
    path = tmp_path / "runs.jsonl"
    log = RunLog(path, held_out_ids={"q1"})
    configs = [
        batch_config(name, "held_out")
        for name in ("v1", "v2a", "v2b", "v2b_norerank")
    ]
    configs[-1] = RunConfig(
        "v2b_norerank", "missing-parent", 512, 64, 2, 50, 60, "prompt", "held_out", generation_k=2
    )

    with pytest.raises(ValueError, match="parent run ID"):
        run_held_out_once([question("held_out")], pipeline, configs, log)

    assert not path.exists() or path.read_text(encoding="utf-8") == ""


    fake_pipeline = FakePipeline()
    pipeline = type(
        "Pipeline",
        (),
        {
            "final_k": 2,
            "retrievers": {"v1": FakeRetriever()},
            "ask": lambda self, q, config, doc_ids=None, hits=None: fake_pipeline.ask(
                q, config, doc_ids
            ),
        },
    )()
    result = run_config(
        "v1",
        [question()],
        pipeline,
        run_config_fixture(),
        RunLog(tmp_path / "runs.jsonl"),
    )
    assert result[0].answer == "Answer"
    assert result[0].retrieved_chunk_ids == ["doc:0", "doc:1"]
    assert result[0].retrieval.recall_at_k == 1.0


def test_score_run_keeps_one_score_per_sample(monkeypatch):
    results = [
        type(
            "Result", (), {"question": question(), "answer": "A", "contexts": ["ctx"]}
        )()
    ]
    captured = {}

    def fake_score(samples, cfg, settings):
        captured["samples"] = samples
        captured["settings"] = settings
        return [SampleScores(None, 0.7, 0.4, 0.3)]

    monkeypatch.setattr("docsqa.eval.runner.score_samples", fake_score)
    assert score_run(results, [JudgeConfig("local", "local", "model")]) == {
        "local": [SampleScores(None, 0.7, 0.4, 0.3)]
    }
    assert len(captured["samples"]) == 1
