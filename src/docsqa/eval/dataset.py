from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal


Category = Literal["fact", "reasoning", "multi_chunk", "rare_keyword", "out_of_scope"]
Split = Literal["dev", "held_out"]
CATEGORIES: tuple[Category, ...] = (
    "fact",
    "reasoning",
    "multi_chunk",
    "rare_keyword",
    "out_of_scope",
)


class DatasetError(ValueError):
    """Raised when the evaluation dataset violates its schema or protocol."""


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    ground_truth: str
    category: Category
    gold_pages: list[int]
    gold_spans: list[str]
    author_date: str
    split: Split

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise DatasetError(f"unknown category: {self.category}")
        if self.split not in ("dev", "held_out"):
            raise DatasetError(f"unknown split: {self.split}")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Question:
        try:
            return cls(
                id=str(data["id"]),
                text=str(data["text"]),
                ground_truth=str(data["ground_truth"]),
                category=data["category"],  # type: ignore[arg-type]
                gold_pages=list(data.get("gold_pages", [])),  # type: ignore[arg-type]
                gold_spans=list(data.get("gold_spans", [])),  # type: ignore[arg-type]
                author_date=str(data["author_date"]),
                split=data["split"],  # type: ignore[arg-type]
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, DatasetError):
                raise
            raise DatasetError(f"invalid question record: {error}") from error

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "text": self.text,
            "ground_truth": self.ground_truth,
            "category": self.category,
            "gold_pages": self.gold_pages,
            "gold_spans": self.gold_spans,
            "author_date": self.author_date,
            "split": self.split,
        }


def load_questions(path: Path) -> list[Question]:
    questions: list[Question] = []
    try:
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DatasetError(
                        f"invalid JSON on line {line_number}: {error.msg}"
                    ) from error
                if not isinstance(value, dict):
                    raise DatasetError(f"line {line_number} must contain a JSON object")
                questions.append(Question.from_dict(value))
    except OSError as error:
        raise DatasetError(f"could not read dataset {path}") from error
    return questions


def validate_dataset(questions: list[Question]) -> None:
    if not 40 <= len(questions) <= 60:
        raise DatasetError("dataset must contain 40 to 60 questions")

    ids = [question.id for question in questions]
    if any(not question_id.strip() for question_id in ids):
        raise DatasetError("question IDs must not be empty")
    if len(ids) != len(set(ids)):
        raise DatasetError("question IDs must be unique")

    held_out_count = sum(question.split == "held_out" for question in questions)
    if not 15 <= held_out_count <= 20:
        raise DatasetError("dataset must contain 15 to 20 held_out questions")

    categories = {question.category for question in questions}
    if categories != set(CATEGORIES):
        missing = sorted(set(CATEGORIES) - categories)
        raise DatasetError(f"dataset is missing required categories: {', '.join(missing)}")

    out_of_scope_count = sum(
        question.category == "out_of_scope" for question in questions
    )
    if out_of_scope_count < 6:
        raise DatasetError("dataset must contain at least 6 out-of-scope questions")

    for question in questions:
        if not question.id.strip() or not question.text.strip() or not question.ground_truth.strip():
            raise DatasetError(f"question {question.id!r} has an empty required field")
        if question.category == "out_of_scope":
            if question.gold_pages or question.gold_spans:
                raise DatasetError(
                    f"out-of-scope question {question.id!r} must not have gold evidence"
                )
            continue
        if not question.gold_pages and not question.gold_spans:
            raise DatasetError(
                f"in-scope question {question.id!r} requires gold evidence"
            )
        if any(page < 1 for page in question.gold_pages):
            raise DatasetError(f"question {question.id!r} has an invalid gold page")
        if any(
            question.ground_truth.strip() == span.strip()
            for span in question.gold_spans
        ):
            raise DatasetError(
                f"ground truth for question {question.id!r} must not be a verbatim gold span"
            )

    category_splits: dict[Category, Counter[str]] = {
        category: Counter(
            question.split for question in questions if question.category == category
        )
        for category in CATEGORIES
    }
    for category, counts in category_splits.items():
        if not counts["dev"] or not counts["held_out"]:
            raise DatasetError(
                f"split must be stratified: category {category!r} must appear in dev and held_out"
            )


def _split_payload(questions: list[Question]) -> dict[str, list[str]]:
    return {
        split: sorted(question.id for question in questions if question.split == split)
        for split in ("dev", "held_out")
    }


def _split_hash(splits: dict[str, list[str]]) -> str:
    serialized = json.dumps(splits, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def freeze_split(questions: list[Question], path: Path) -> str:
    validate_dataset(questions)
    splits = _split_payload(questions)
    fingerprint = _split_hash(splits)
    payload = {"splits": splits, "sha256": fingerprint}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return fingerprint


def verify_split(questions: list[Question], path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        splits = payload["splits"]
        recorded_hash = payload["sha256"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise DatasetError(f"invalid frozen split file: {path}") from error

    if not isinstance(splits, dict) or set(splits) != {"dev", "held_out"}:
        raise DatasetError("frozen split must contain dev and held_out IDs")
    expected_hash = _split_hash(splits)
    if recorded_hash != expected_hash:
        raise DatasetError("frozen split hash does not match its IDs")
    if splits != _split_payload(questions):
        raise DatasetError("question IDs or split assignments differ from frozen split")
