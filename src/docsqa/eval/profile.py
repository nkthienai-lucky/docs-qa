from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from ..pipeline import Pipeline
from .dataset import Question


@dataclass(frozen=True)
class ProfileSample:
    question_id: str
    config: str
    retrieval_s: float
    generation_s: float
    total_s: float
    retrieved_chunk_ids: list[str]
    abstained: bool


def profile_config(
    config_name: str,
    questions: list[Question],
    pipeline: Pipeline,
    *,
    out_path: Path,
) -> list[ProfileSample]:
    """Measure dev retrieval and answer-generation latency without audit logging."""
    if any(question.split != "dev" for question in questions):
        raise ValueError("profiling is limited to dev questions")
    if config_name not in pipeline.retrievers:
        raise ValueError(f"unknown config: {config_name}")

    samples = []
    retriever = pipeline.retrievers[config_name]
    for question in questions:
        started = perf_counter()
        retrieval_started = perf_counter()
        hits = retriever.retrieve(question.text)[: pipeline.final_k]
        retrieval_s = perf_counter() - retrieval_started
        generation_started = perf_counter()
        response = pipeline.ask(question.text, config=config_name, hits=hits)
        generation_s = perf_counter() - generation_started
        samples.append(
            ProfileSample(
                question_id=question.id,
                config=config_name,
                retrieval_s=retrieval_s,
                generation_s=generation_s,
                total_s=perf_counter() - started,
                retrieved_chunk_ids=[hit.chunk.chunk_id for hit in hits],
                abstained=response.abstained,
            )
        )

    payload = {
        "kind": "dev_profiling_only",
        "config": config_name,
        "samples": [asdict(sample) for sample in samples],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return samples


def write_profile_csv(samples: list[ProfileSample], out_path: Path) -> None:
    fields = [
        "question_id",
        "config",
        "retrieval_s",
        "generation_s",
        "total_s",
        "retrieved_chunk_ids",
        "abstained",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            row: dict[str, Any] = asdict(sample)
            row["retrieved_chunk_ids"] = ";".join(sample.retrieved_chunk_ids)
            writer.writerow(row)
