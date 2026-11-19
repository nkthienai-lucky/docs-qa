from __future__ import annotations

from pathlib import Path

import pytest

from docsqa.eval.dataset import Question
from docsqa.eval.judges import SampleScores
from docsqa.eval.metrics import RetrievalScores
from docsqa.eval.report import build_report, write_report
from docsqa.eval.runner import SampleResult


def sample(question_id, category, split, abstained, recall, score=0.8):
    question = Question(
        question_id,
        f"Question {question_id}",
        "Ground truth",
        category,
        [] if category == "out_of_scope" else [1],
        [] if category == "out_of_scope" else ["Evidence"],
        "2026-11-17",
        split,
    )
    return SampleResult(
        question=question,
        config="v1",
        answer="Answer",
        abstained=abstained,
        retrieved_chunk_ids=["doc:0"],
        retrieval_scores=[0.9],
        retrieval=RetrievalScores(recall, 1.0, 1.0),
        contexts=["Evidence"],
    )


def test_out_of_scope_excluded_from_all_four_ragas_means_but_scored_for_refusal():
    dev = [
        sample("in", "fact", "dev", False, 1),
        sample("oos", "out_of_scope", "dev", True, 0),
    ]
    scores = {"local": [SampleScores(0.8, 0.7, 0.6, 0.5)]}
    report = build_report(dev, [], "local", scores=scores)
    assert report.metrics["v1"]["local"]["faithfulness"] == 0.8
    assert report.refusal_accuracy["dev"]["v1"] == 1.0
    assert report.false_refusal_rate["dev"]["v1"] == 0.0


def test_false_refusal_rate_counts_abstains_on_answerable_questions():
    dev = [
        sample("in", "fact", "dev", True, 1),
        sample("oos", "out_of_scope", "dev", True, 0),
    ]
    report = build_report(dev, [], "local")
    assert report.false_refusal_rate["dev"]["v1"] == 1.0


def test_better_requires_both_judges_agree_and_ci_excludes_zero_else_inconclusive():
    questions = [sample(str(i), "fact", "dev", False, 1) for i in range(20)]
    results = [
        result
        for question in questions
        for result in (
            question,
            SampleResult(**{**question.__dict__, "config": "v2a"}),
        )
    ]
    score_map = {
        "api": [
            score
            for _ in questions
            for score in (
                SampleScores(0.8, 0.8, 0.8, 0.8),
                SampleScores(0.9, 0.8, 0.8, 0.8),
            )
        ],
        "local": [
            score
            for _ in questions
            for score in (
                SampleScores(0.7, 0.7, 0.7, 0.7),
                SampleScores(0.8, 0.7, 0.7, 0.7),
            )
        ],
    }
    report = build_report(
        results, [], "api", scores=score_map, configs={"v1": "v1", "v2a": "v2a"}
    )
    comparison = report.comparisons["v2a-v1"]["faithfulness"]
    assert comparison["verdict"] == "better"

    score_map["local"] = [
        score
        for _ in questions
        for score in (
            SampleScores(0.8, 0.7, 0.7, 0.7),
            SampleScores(0.7, 0.7, 0.7, 0.7),
        )
    ]
    report = build_report(
        results, [], "api", scores=score_map, configs={"v1": "v1", "v2a": "v2a"}
    )
    assert report.comparisons["v2a-v1"]["faithfulness"]["verdict"] == "inconclusive"


def test_report_keeps_dev_and_held_out_metrics_and_comparisons_separate():
    dev = [
        result
        for index in range(4)
        for result in (
            sample(f"dev-{index}", "fact", "dev", False, 1),
            SampleResult(
                **{
                    **sample(f"dev-{index}", "fact", "dev", False, 1).__dict__,
                    "config": "v2a",
                }
            ),
        )
    ]
    held_out = [
        result
        for index in range(4)
        for result in (
            sample(f"held-{index}", "fact", "held_out", False, 1),
            SampleResult(
                **{
                    **sample(f"held-{index}", "fact", "held_out", False, 1).__dict__,
                    "config": "v2a",
                }
            ),
        )
    ]
    scores = {
        "primary": [
            score
            for index in range(4)
            for score in (
                SampleScores(0.8, 0.8, 0.8, 0.8),
                SampleScores(0.9, 0.8, 0.8, 0.9),
            )
        ]
        + [
            score
            for index in range(4)
            for score in (
                SampleScores(0.6, 0.6, 0.6, 0.6),
                SampleScores(0.7, 0.6, 0.6, 0.7),
            )
        ]
    }
    report = build_report(
        dev, held_out, "primary", scores=scores, configs={"v1": "v1", "v2a": "v2a"}
    )
    assert report.metrics_by_split["dev"]["v1"]["primary"][
        "faithfulness"
    ] == pytest.approx(0.8)
    assert report.metrics_by_split["dev"]["v2a"]["primary"][
        "faithfulness"
    ] == pytest.approx(0.9)
    assert report.metrics_by_split["held_out"]["v1"]["primary"][
        "faithfulness"
    ] == pytest.approx(0.6)
    assert report.metrics_by_split["held_out"]["v2a"]["primary"][
        "faithfulness"
    ] == pytest.approx(0.7)
    assert report.metrics["v1"]["primary"]["faithfulness"] == pytest.approx(0.6)
    assert report.metrics["v2a"]["primary"]["faithfulness"] == pytest.approx(0.7)
    assert (
        report.comparisons_by_split["dev"]["v2a-v1"]["faithfulness"]["verdict"]
        == "inconclusive"
    )
    assert (
        report.comparisons_by_split["held_out"]["v2a-v1"]["faithfulness"]["verdict"]
        == "inconclusive"
    )
    assert report.thresholds["faithfulness"] is False
    assert report.category_counts["dev"]["v1"]["fact"] == 4
    assert report.category_counts["held_out"]["v2a"]["fact"] == 4
    assert report.category_scores["dev"]["v2a"]["fact"]["primary"][
        "faithfulness"
    ] == pytest.approx(0.9)
    assert (
        report.retrieval_comparisons_by_split["dev"]["v2a-v1"]["recall_at_k"]["verdict"]
        == "no_detectable_difference"
    )


def test_retrieval_comparison_reports_paired_ci():
    results = []
    for index in range(20):
        baseline = sample(f"q{index}", "fact", "dev", False, 0)
        improved = SampleResult(
            **{**sample(f"q{index}", "fact", "dev", False, 1).__dict__, "config": "v2a"}
        )
        results.extend((baseline, improved))
    report = build_report(results, [], "primary", configs={"v1": "v1", "v2a": "v2a"})
    comparison = report.retrieval_comparisons_by_split["dev"]["v2a-v1"]["recall_at_k"]
    assert comparison["mean_diff"] == pytest.approx(1.0)
    assert comparison["verdict"] == "better"


def test_report_csv_exports_retrieval_and_judge_scores(tmp_path: Path):
    results = [sample("q1", "fact", "dev", False, 1)]
    scores = {"primary": [SampleScores(0.9, 0.8, 0.7, 0.6)]}
    report = build_report(results, [], "primary", scores=scores)
    write_report(report, tmp_path)
    csv_text = (tmp_path / "per-question.csv").read_text(encoding="utf-8")
    assert "retrieval_scores" in csv_text
    assert "judge_scores" in csv_text
    assert "context_precision" in csv_text
    assert "0.9" in csv_text


def test_judge_correlation_aligns_same_config_by_question_and_excludes_out_of_scope():
    results = [
        sample("q1", "fact", "dev", False, 1),
        SampleResult(
            **{**sample("q1", "fact", "dev", False, 1).__dict__, "config": "v2a"}
        ),
        sample("q2", "fact", "dev", False, 1),
        SampleResult(
            **{**sample("q2", "fact", "dev", False, 1).__dict__, "config": "v2a"}
        ),
        sample("oos", "out_of_scope", "dev", True, 0),
        SampleResult(
            **{
                **sample("oos", "out_of_scope", "dev", True, 0).__dict__,
                "config": "v2a",
            }
        ),
    ]
    scores = {
        "first": [
            SampleScores(0.1, 0.1, 0.1, 0.1),
            SampleScores(0.9, 0.9, 0.9, 0.9),
            SampleScores(0.2, 0.2, 0.2, 0.2),
            SampleScores(0.8, 0.8, 0.8, 0.8),
        ],
        "second": [
            SampleScores(0.9, 0.9, 0.9, 0.9),
            SampleScores(0.1, 0.1, 0.1, 0.1),
            SampleScores(0.8, 0.8, 0.8, 0.8),
            SampleScores(0.2, 0.2, 0.2, 0.2),
        ],
    }
    report = build_report(
        results,
        [],
        "first",
        scores=scores,
        configs={"v1": "v1", "v2a": "v2a"},
    )
    assert report.judge_correlation == pytest.approx(-1.0)


def test_threshold_met_only_when_ci_lower_bound_above_marker():
    results = [sample(str(i), "fact", "held_out", False, 1) for i in range(20)]
    scores = {"primary": [SampleScores(1.0, 0.8, 0.8, 1.0) for _ in results]}
    report = build_report([], results, "primary", scores=scores)
    assert report.thresholds["faithfulness"] is True
    scores["primary"] = [SampleScores(0.8, 0.8, 0.8, 0.7) for _ in results]
    report = build_report([], results, "primary", scores=scores)
    assert report.thresholds["faithfulness"] is False


def test_repeat_variance_uses_per_run_mean_scores():
    results = [sample("q1", "fact", "dev", False, 1)]
    report = build_report(
        results,
        [],
        "primary",
        repeated_scores={
            "v1": [
                [SampleScores(0.2, 0.2, 0.2, 0.2), SampleScores(0.4, 0.4, 0.4, 0.4)],
                [SampleScores(0.6, 0.6, 0.6, 0.6), SampleScores(0.8, 0.8, 0.8, 0.8)],
            ]
        },
    )
    assert report.repeat_variance["v1"] == pytest.approx(0.08)


def test_default_config_falls_back_to_v1_when_intervals_overlap():
    results = [sample(str(i), "fact", "dev", False, 1) for i in range(20)]
    scores = {"primary": [SampleScores(0.8, 0.8, 0.8, 0.8) for _ in results]}
    report = build_report(results, [], "primary", scores=scores)
    assert report.default_config == "v1"


def test_default_config_uses_primary_judge_dev_comparisons_with_one_judge():
    results = []
    for index in range(20):
        baseline = sample(str(index), "fact", "dev", False, 1)
        improved = SampleResult(
            **{
                **sample(str(index), "fact", "dev", False, 1).__dict__,
                "config": "v2a",
            }
        )
        results.extend((baseline, improved))
    scores = {
        "api": [
            score
            for _ in range(20)
            for score in (
                SampleScores(0.6, 0.6, 0.6, 0.6),
                SampleScores(0.9, 0.9, 0.9, 0.9),
            )
        ]
    }

    report = build_report(
        results,
        [],
        "api",
        scores=scores,
        configs={"v1": "v1", "v2a": "v2a"},
    )

    assert report.default_config == "v2a"


def test_default_config_selection_uses_dev_not_held_out():
    dev = [sample(str(index), "fact", "dev", False, 1) for index in range(20)]
    held_out = [sample(str(index), "fact", "held_out", False, 1) for index in range(20)]
    scores = {
        "api": [SampleScores(0.8, 0.8, 0.8, 0.8) for _ in dev + held_out]
    }

    report = build_report(dev, held_out, "api", scores=scores)

    assert report.default_config == "v1"


def test_report_files_contain_no_secret_values_or_gateway_host(tmp_path: Path):
    results = [sample("q1", "fact", "dev", False, 1)]
    report = build_report(results, [], "primary")
    write_report(report, tmp_path, forbidden=["secret-token", "gateway.private.test"])
    contents = "".join(path.read_text(encoding="utf-8") for path in tmp_path.iterdir())
    assert "secret-token" not in contents
    assert "gateway.private.test" not in contents
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "per-question.csv").exists()
    assert (tmp_path / "metadata.json").exists()


def test_final_rerun_report_names_v2a_as_default_and_keeps_caveat():
    report_path = (
        Path(__file__).parents[2]
        / "eval/protocol-v2/report-final-rerun-gpu-api/report.md"
    )
    text = report_path.read_text(encoding="utf-8")

    assert "Default configuration: `v2a`" in text
    assert "Default configuration: `v1`" not in text
    assert "v2b_norerank" in text
    assert "post-interruption rerun" in text
