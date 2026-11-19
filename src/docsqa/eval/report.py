from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, variance
from typing import Any

from .judges import HumanLabel, SampleScores, agreement_with_labels
from .runlog import assert_no_secrets, build_metadata
from .runner import SampleResult
from .stats import paired_bootstrap_ci, spearman, verdict


@dataclass(frozen=True)
class Report:
    metrics: dict[str, dict[str, dict[str, float | None]]]
    metrics_by_split: dict[str, dict[str, dict[str, dict[str, float | None]]]]
    retrieval_by_split: dict[str, dict[str, dict[str, float | None]]]
    retrieval_comparisons_by_split: dict[str, dict[str, dict[str, Any]]]
    category_counts: dict[str, dict[str, dict[str, int]]]
    category_scores: dict[str, dict[str, dict[str, dict[str, dict[str, float | None]]]]]
    refusal_accuracy: dict[str, dict[str, float | None]]
    false_refusal_rate: dict[str, dict[str, float | None]]
    comparisons: dict[str, dict[str, dict[str, Any]]]
    comparisons_by_split: dict[str, dict[str, dict[str, dict[str, Any]]]]
    judge_correlation: float | None
    judge_agreement: dict[str, float | None]
    nan_counts: dict[str, dict[str, int]]
    repeat_variance: dict[str, float | None]
    thresholds: dict[str, bool | None]
    thresholds_by_config: dict[str, dict[str, bool | None]]
    default_config: str
    label: str
    per_question: list[dict[str, Any]]
    metadata: dict[str, Any]


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _score_rows(
    results: list[SampleResult], scores: dict[str, list[SampleScores]]
) -> dict[str, list[SampleScores]]:
    in_scope_count = sum(
        result.question.category != "out_of_scope" for result in results
    )
    normalized = {}
    for judge, judge_scores in scores.items():
        if len(judge_scores) != in_scope_count:
            raise ValueError(f"judge {judge!r} must have one score per in-scope sample")
        expanded = []
        score_index = 0
        for result in results:
            if result.question.category == "out_of_scope":
                expanded.append(SampleScores(None, None, None, None))
            else:
                expanded.append(judge_scores[score_index])
                score_index += 1
        normalized[judge] = expanded
    return normalized


def _mean_metric(scores: list[SampleScores], field: str) -> float | None:
    values = [getattr(score, field) for score in scores]
    return _mean(
        [value for value in values if value is not None and math.isfinite(value)]
    )


def _refusal_metrics(
    results: list[SampleResult], split: str
) -> tuple[float | None, float | None]:
    scoped = [result for result in results if result.question.split == split]
    out_of_scope = [
        result for result in scoped if result.question.category == "out_of_scope"
    ]
    answerable = [
        result for result in scoped if result.question.category != "out_of_scope"
    ]
    accuracy = (
        sum(result.abstained for result in out_of_scope) / len(out_of_scope)
        if out_of_scope
        else None
    )
    false_refusal = (
        sum(result.abstained for result in answerable) / len(answerable)
        if answerable
        else None
    )
    return accuracy, false_refusal


def _retrieval_means(results: list[SampleResult]) -> dict[str, float | None]:
    relevant = [result.retrieval for result in results if result.retrieval is not None]
    return {
        field: _mean([getattr(score, field) for score in relevant])
        for field in ("recall_at_k", "precision_at_k", "reciprocal_rank")
    }


def _comparison(
    first: list[tuple[str, SampleScores]],
    second: list[tuple[str, SampleScores]],
    field: str,
) -> dict[str, Any]:
    first_by_id = dict(first)
    second_by_id = dict(second)
    pairs = []
    for question_id in sorted(first_by_id.keys() & second_by_id.keys()):
        left = getattr(first_by_id[question_id], field)
        right = getattr(second_by_id[question_id], field)
        if (
            left is not None
            and right is not None
            and math.isfinite(left)
            and math.isfinite(right)
        ):
            pairs.append((left, right))
    if len(pairs) < 2:
        return {"mean_diff": None, "ci": [None, None], "verdict": "inconclusive"}
    difference, lower, upper = paired_bootstrap_ci(
        [left for left, _ in pairs], [right for _, right in pairs], seed=0
    )
    return {
        "mean_diff": difference,
        "ci": [lower, upper],
        "verdict": verdict(lower, upper),
    }


def _retrieval_comparison(
    first: list[SampleResult], second: list[SampleResult], field: str
) -> dict[str, Any]:
    first_by_id = {result.question.id: result for result in first}
    second_by_id = {result.question.id: result for result in second}
    pairs = [
        (
            getattr(first_by_id[qid].retrieval, field),
            getattr(second_by_id[qid].retrieval, field),
        )
        for qid in sorted(first_by_id.keys() & second_by_id.keys())
        if first_by_id[qid].retrieval is not None
        and second_by_id[qid].retrieval is not None
    ]
    if len(pairs) < 2:
        return {"mean_diff": None, "ci": [None, None], "verdict": "inconclusive"}
    difference, lower, upper = paired_bootstrap_ci(
        [left for left, _ in pairs], [right for _, right in pairs], seed=0
    )
    return {
        "mean_diff": difference,
        "ci": [lower, upper],
        "verdict": verdict(lower, upper),
    }


def build_report(
    dev_results: list[SampleResult],
    held_out_results: list[SampleResult],
    primary_judge: str,
    *,
    scores: dict[str, list[SampleScores]] | None = None,
    configs: dict[str, str] | None = None,
    labels: list[HumanLabel] | None = None,
    repeated_scores: dict[str, list[list[SampleScores]]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Report:
    all_results = dev_results + held_out_results
    scores = scores or {}
    normalized_scores = _score_rows(all_results, scores) if scores else {}
    config_names = configs or {result.config: result.config for result in all_results}
    fields = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    retrieval_fields = ("recall_at_k", "precision_at_k", "reciprocal_rank")

    headline_split = "held_out" if held_out_results else "dev"
    metrics = {
        config_name: {
            judge: {
                field: _mean_metric(
                    [
                        values[index]
                        for index, result in enumerate(all_results)
                        if result.config == actual_name
                        and result.question.split == headline_split
                    ],
                    field,
                )
                for field in fields
            }
            for judge, values in normalized_scores.items()
        }
        for config_name, actual_name in config_names.items()
    }
    nan_counts = {
        judge: {
            field: sum(
                getattr(score, field) is None
                or not math.isfinite(getattr(score, field))
                for score in judge_scores
            )
            for field in fields
        }
        for judge, judge_scores in normalized_scores.items()
    }

    comparison_pairs = (
        ("v2a-v1", "v2a", "v1"),
        ("v2b-v1", "v2b", "v1"),
        ("v2b-v2a", "v2b", "v2a"),
        ("v2b_norerank-v1", "v2b_norerank", "v1"),
    )
    comparisons_by_split: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    retrieval_comparisons_by_split: dict[str, dict[str, dict[str, Any]]] = {}
    metrics_by_split: dict[str, dict[str, dict[str, dict[str, float | None]]]] = {}
    retrieval_by_split: dict[str, dict[str, dict[str, float | None]]] = {}

    for split in ("dev", "held_out"):
        split_results = [
            result for result in all_results if result.question.split == split
        ]
        metrics_by_split[split] = {}
        retrieval_by_split[split] = {}
        grouped: dict[str, dict[str, list[tuple[str, SampleScores]]]] = {}
        result_groups: dict[str, list[SampleResult]] = {}
        for config_name, actual_name in config_names.items():
            config_results = [
                result for result in split_results if result.config == actual_name
            ]
            indices = [
                index
                for index, result in enumerate(all_results)
                if result.question.split == split and result.config == actual_name
            ]
            metrics_by_split[split][config_name] = {
                judge: {
                    field: _mean_metric([values[index] for index in indices], field)
                    for field in fields
                }
                for judge, values in normalized_scores.items()
            }
            retrieval_by_split[split][config_name] = _retrieval_means(config_results)
            result_groups[config_name] = config_results
            for judge, values in normalized_scores.items():
                grouped.setdefault(config_name, {})[judge] = [
                    (all_results[index].question.id, values[index]) for index in indices
                ]

        split_comparisons: dict[str, dict[str, dict[str, Any]]] = {}
        split_retrieval_comparisons: dict[str, dict[str, Any]] = {}
        for comparison_name, candidate, baseline in comparison_pairs:
            if candidate not in result_groups or baseline not in result_groups:
                continue
            if candidate in grouped and baseline in grouped:
                split_comparisons[comparison_name] = {}
                for field in fields:
                    per_judge = {
                        judge: _comparison(
                            grouped[candidate][judge], grouped[baseline][judge], field
                        )
                        for judge in grouped[candidate]
                    }
                    verdicts = [item["verdict"] for item in per_judge.values()]
                    shared_verdict = (
                        verdicts[0]
                        if len(verdicts) == 2
                        and verdicts[0] in {"better", "worse"}
                        and all(value == verdicts[0] for value in verdicts)
                        else "inconclusive"
                    )
                    split_comparisons[comparison_name][field] = {
                        "judges": per_judge,
                        "verdict": shared_verdict,
                    }
            split_retrieval_comparisons[comparison_name] = {}
            for field in retrieval_fields:
                split_retrieval_comparisons[comparison_name][field] = (
                    _retrieval_comparison(
                        result_groups[candidate], result_groups[baseline], field
                    )
                )
        comparisons_by_split[split] = split_comparisons
        retrieval_comparisons_by_split[split] = split_retrieval_comparisons

    headline_split = "held_out" if held_out_results else "dev"
    comparisons = comparisons_by_split[headline_split]
    refusal_accuracy: dict[str, dict[str, float | None]] = {}
    false_refusal_rate: dict[str, dict[str, float | None]] = {}
    for split in ("dev", "held_out"):
        refusal_accuracy[split] = {}
        false_refusal_rate[split] = {}
        for config_name, actual_name in config_names.items():
            selected = [
                result
                for result in all_results
                if result.question.split == split and result.config == actual_name
            ]
            (
                refusal_accuracy[split][config_name],
                false_refusal_rate[split][config_name],
            ) = _refusal_metrics(selected, split)

    categories = sorted({result.question.category for result in all_results})
    category_counts = {
        split: {
            config_name: {
                category: sum(
                    result.question.split == split
                    and result.config == actual_name
                    and result.question.category == category
                    for result in all_results
                )
                for category in categories
            }
            for config_name, actual_name in config_names.items()
        }
        for split in ("dev", "held_out")
    }
    category_scores = {}
    for split in ("dev", "held_out"):
        category_scores[split] = {}
        for config_name, actual_name in config_names.items():
            category_scores[split][config_name] = {}
            for category in categories:
                indices = [
                    index
                    for index, result in enumerate(all_results)
                    if result.question.split == split
                    and result.config == actual_name
                    and result.question.category == category
                ]
                category_scores[split][config_name][category] = {
                    judge: {
                        field: _mean_metric([values[index] for index in indices], field)
                        if category != "out_of_scope"
                        else None
                        for field in fields
                    }
                    for judge, values in normalized_scores.items()
                }
    category_scores = {
        split: config_values for split, config_values in category_scores.items()
    }

    judge_correlation = None
    judge_names = list(normalized_scores)
    if len(judge_names) >= 2:
        preferred_config = (
            "v1" if "v1" in config_names else next(iter(config_names), None)
        )
        actual_config = config_names.get(preferred_config) if preferred_config else None
        aligned: dict[str, dict[str, float]] = {}
        for index, result in enumerate(all_results):
            if (
                result.config != actual_config
                or result.question.category == "out_of_scope"
            ):
                continue
            for judge in judge_names[:2]:
                value = normalized_scores[judge][index].faithfulness
                if value is not None and math.isfinite(value):
                    aligned.setdefault(result.question.id, {})[judge] = value
        pairs = [
            (values[judge_names[0]], values[judge_names[1]])
            for values in aligned.values()
            if all(judge in values for judge in judge_names[:2])
        ]
        if len(pairs) >= 2:
            judge_correlation = spearman(
                [left for left, _ in pairs], [right for _, right in pairs]
            )

    judge_agreement: dict[str, float | None] = {}
    if labels:
        label_ids = [label.sample_id for label in labels]
        if len(set(label_ids)) != len(label_ids):
            raise ValueError("human label sample IDs must be unique")
        preferred_config = (
            "v1" if "v1" in config_names else next(iter(config_names), None)
        )
        actual_config = config_names.get(preferred_config) if preferred_config else None
        result_by_id = {
            result.question.id: index
            for index, result in enumerate(all_results)
            if result.config == actual_config
        }
        indices = [result_by_id.get(question_id) for question_id in label_ids]
        for judge, values in normalized_scores.items():
            if any(index is None for index in indices):
                judge_agreement[judge] = None
            else:
                judge_agreement[judge] = agreement_with_labels(
                    [values[index] for index in indices if index is not None], labels
                )

    thresholds_by_config: dict[str, dict[str, bool | None]] = {}
    for config_name, actual_name in config_names.items():
        indices = [
            index
            for index, result in enumerate(all_results)
            if result.question.split == "held_out" and result.config == actual_name
        ]
        thresholds_by_config[config_name] = {}
        for key, field, marker in (
            ("faithfulness", "faithfulness", 0.8),
            ("context_recall", "context_recall", 0.7),
        ):
            values = [
                getattr(normalized_scores[primary_judge][index], field)
                for index in indices
                if primary_judge in normalized_scores
                and getattr(normalized_scores[primary_judge][index], field) is not None
                and math.isfinite(
                    getattr(normalized_scores[primary_judge][index], field)
                )
            ]
            if len(values) >= 2:
                _, lower, _ = paired_bootstrap_ci(
                    values, [marker] * len(values), seed=0
                )
                thresholds_by_config[config_name][key] = lower > 0
            else:
                thresholds_by_config[config_name][key] = None
    thresholds = thresholds_by_config.get(
        "v1", next(iter(thresholds_by_config.values()), {})
    )

    default_config = "v1"
    primary_comparisons = comparisons_by_split.get("dev", {})
    for candidate in ("v2b", "v2a"):
        candidate_comparisons = [
            comparison
            for name, comparison in primary_comparisons.items()
            if name.startswith(candidate + "-")
        ]
        if not candidate_comparisons:
            continue
        primary_verdicts = [
            metric.get("judges", {}).get(primary_judge, {}).get("verdict")
            for comparison in candidate_comparisons
            for metric in comparison.values()
        ]
        if primary_verdicts and all(verdict == "better" for verdict in primary_verdicts):
            default_config = candidate
            break

    per_question = []
    for index, result in enumerate(all_results):
        per_question.append(
            {
                "question_id": result.question.id,
                "config": result.config,
                "split": result.question.split,
                "category": result.question.category,
                "answer": result.answer,
                "abstained": result.abstained,
                "retrieved_chunk_ids": result.retrieved_chunk_ids,
                "retrieval_scores": result.retrieval_scores,
                "retrieval": asdict(result.retrieval) if result.retrieval else None,
                "contexts": result.contexts,
                "judge_scores": {
                    judge: asdict(values[index])
                    for judge, values in normalized_scores.items()
                },
            }
        )

    repeat_variance = {}
    for config_name, runs in (repeated_scores or {}).items():
        run_means = []
        for run in runs:
            finite = [
                score.faithfulness
                for score in run
                if score.faithfulness is not None and math.isfinite(score.faithfulness)
            ]
            if finite:
                run_means.append(mean(finite))
        repeat_variance[config_name] = (
            variance(run_means) if len(run_means) > 1 else None
        )

    report_metadata = metadata or {}
    status = report_metadata.get("status")
    if status == "retrieval_only":
        label = "dev, retrieval-only; incomplete"
    else:
        label = (
            "held-out headline; dev, optimistic"
            if dev_results and held_out_results
            else ("held-out" if held_out_results else "dev, optimistic")
        )
    return Report(
        metrics=metrics,
        metrics_by_split=metrics_by_split,
        retrieval_by_split=retrieval_by_split,
        retrieval_comparisons_by_split=retrieval_comparisons_by_split,
        category_counts=category_counts,
        category_scores=category_scores,
        refusal_accuracy=refusal_accuracy,
        false_refusal_rate=false_refusal_rate,
        comparisons=comparisons,
        comparisons_by_split=comparisons_by_split,
        judge_correlation=judge_correlation,
        judge_agreement=judge_agreement,
        nan_counts=nan_counts,
        repeat_variance=repeat_variance,
        thresholds=thresholds,
        thresholds_by_config=thresholds_by_config,
        default_config=default_config,
        label=label,
        per_question=per_question,
        metadata=build_metadata(metadata or {}),
    )


def write_report(
    report: Report,
    out_dir: Path,
    *,
    forbidden: list[str] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    assert_no_secrets(asdict(report), forbidden or [])
    (out_dir / "report.md").write_text(
        "# Docs QA evaluation\n\n"
        f"Label: **{report.label}**\n\n"
        f"Default configuration: `{report.default_config}`\n\n"
        "## Metrics by split and configuration\n\n"
        "```json\n"
        + json.dumps(
            {
                "metrics_by_split": report.metrics_by_split,
                "retrieval_by_split": report.retrieval_by_split,
                "retrieval_comparisons_by_split": report.retrieval_comparisons_by_split,
                "comparisons_by_split": report.comparisons_by_split,
                "category_counts": report.category_counts,
                "category_scores": report.category_scores,
                "refusal_accuracy": report.refusal_accuracy,
                "false_refusal_rate": report.false_refusal_rate,
                "judge_correlation": report.judge_correlation,
                "judge_agreement": report.judge_agreement,
                "nan_counts": report.nan_counts,
                "repeat_variance": report.repeat_variance,
                "thresholds_by_config": report.thresholds_by_config,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n```\n",
        encoding="utf-8",
    )
    fields = [
        "question_id",
        "config",
        "split",
        "category",
        "answer",
        "abstained",
        "retrieved_chunk_ids",
        "retrieval_scores",
        "retrieval",
        "contexts",
        "judge_scores",
    ]
    json_fields = {
        "retrieved_chunk_ids",
        "retrieval_scores",
        "retrieval",
        "contexts",
        "judge_scores",
    }
    with (out_dir / "per-question.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in report.per_question:
            writer.writerow(
                {
                    field: json.dumps(row[field], ensure_ascii=False)
                    if field in json_fields
                    else row[field]
                    for field in fields
                }
            )
    (out_dir / "metadata.json").write_text(
        json.dumps(report.metadata, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    assert_no_secrets(
        [
            path.read_text(encoding="utf-8")
            for path in out_dir.iterdir()
            if path.is_file()
        ],
        forbidden or [],
    )
