from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from docsqa.eval.dataset import (
    DatasetError,
    Question,
    freeze_split,
    load_questions,
    validate_dataset,
    verify_split,
)

CATEGORIES = ("fact", "reasoning", "multi_chunk", "rare_keyword", "out_of_scope")


def make_dataset() -> list[Question]:
    questions = []
    for index in range(40):
        category = CATEGORIES[index % len(CATEGORIES)]
        held_out = index < 15
        questions.append(
            Question(
                id=f"q{index:02}",
                text=f"Question {index}",
                ground_truth=f"Ground truth {index}",
                category=category,
                gold_pages=[] if category == "out_of_scope" else [index + 1],
                gold_spans=[] if category == "out_of_scope" else [f"Evidence {index}"],
                author_date="2026-11-17",
                split="held_out" if held_out else "dev",
            )
        )
    return questions


def test_validate_requires_40_to_60_questions_and_15_to_20_held_out() -> None:
    questions = make_dataset()

    validate_dataset(questions)

    with pytest.raises(DatasetError, match="40 to 60"):
        validate_dataset(questions[:-1])
    with pytest.raises(DatasetError, match="15 to 20"):
        validate_dataset([replace(q, split="dev") for q in questions])


def test_validate_requires_at_least_6_out_of_scope_and_all_categories() -> None:
    questions = make_dataset()
    assert sum(q.category == "out_of_scope" for q in questions) >= 6

    too_few_oos = [
        replace(q, category="fact", gold_pages=[1], gold_spans=["Evidence"])
        if q.category == "out_of_scope" and q.id not in {"q04", "q09", "q14", "q19", "q24"}
        else q
        for q in questions
    ]
    with pytest.raises(DatasetError, match="6 out-of-scope"):
        validate_dataset(too_few_oos)

    missing_category = [
        replace(q, category="fact") if q.category == "reasoning" else q
        for q in questions
    ]
    with pytest.raises(DatasetError, match="categories"):
        validate_dataset(missing_category)


def test_in_scope_requires_gold_evidence_out_of_scope_does_not() -> None:
    questions = make_dataset()
    empty_evidence = [
        replace(q, gold_pages=[], gold_spans=[]) if q.category == "fact" else q
        for q in questions
    ]
    with pytest.raises(DatasetError, match="gold evidence"):
        validate_dataset(empty_evidence)

    validate_dataset(make_dataset())


def test_ground_truth_not_verbatim_copy_of_gold_span() -> None:
    questions = make_dataset()
    questions[0] = replace(questions[0], ground_truth=questions[0].gold_spans[0])

    with pytest.raises(DatasetError, match="verbatim"):
        validate_dataset(questions)


def test_verify_split_fails_if_question_moves_between_splits(tmp_path: Path) -> None:
    questions = make_dataset()
    split_path = tmp_path / "split.json"
    freeze_split(questions, split_path)
    questions[0] = replace(questions[0], split="dev")

    with pytest.raises(DatasetError, match="split"):
        verify_split(questions, split_path)


def test_split_is_stratified_by_category(tmp_path: Path) -> None:
    questions = make_dataset()
    split_path = tmp_path / "split.json"

    fingerprint = freeze_split(questions, split_path)
    data = json.loads(split_path.read_text(encoding="utf-8"))

    assert data["sha256"] == fingerprint
    assert set(data["splits"]) == {"dev", "held_out"}
    for category in CATEGORIES:
        ids = {q.id for q in questions if q.category == category}
        assert ids & set(data["splits"]["dev"])
        assert ids & set(data["splits"]["held_out"])
    assert load_questions
    verify_split(questions, split_path)
