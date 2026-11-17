from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Literal
import uuid


class MultiVariableChange(ValueError):
    """A tuning run changes more than one parameter from its parent."""


class HeldOutReuse(ValueError):
    """The held-out split is recorded more than once."""


class TuningCapExceeded(ValueError):
    """More than three dev tuning runs have been recorded."""


@dataclass(frozen=True)
class RunConfig:
    name: str
    parent_run_id: str | None
    chunk_size: int
    overlap: int
    top_k: int
    candidates: int
    rrf_k: int
    prompt_hash: str
    split: Literal["dev", "held_out"]
    tuning_round: int = 0


_TUNABLE_FIELDS = ("chunk_size", "overlap", "top_k", "candidates", "rrf_k", "prompt_hash")
_METADATA_ALLOWLIST = {
    "git_revision",
    "corpus_fingerprint",
    "model",
    "generator_model",
    "embed_model",
    "embedder_digest",
    "reranker_model",
    "dependency_versions",
    "sampling",
    "num_ctx",
    "prompt_hash",
    "rrf_k",
    "candidate_limits",
    "bm25_tokenizer",
    "pgvector_index_parameters",
    "dataset_hash",
    "split_id",
    "peak_vram_mb",
    "nan_counts",
    "question_ids",
}


def diff_fields(a: RunConfig, b: RunConfig) -> set[str]:
    return {
        field
        for field in _TUNABLE_FIELDS
        if getattr(a, field) != getattr(b, field)
    }


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid run log JSON on line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"run log line {line_number} must be an object")
            records.append(record)
    return records


class RunLog:
    def __init__(self, path: Path, held_out_ids: set[str] | None = None) -> None:
        self.path = path
        self.held_out_ids = held_out_ids or set()

    def record(self, config: RunConfig, metadata: dict[str, Any]) -> str:
        records = _read_records(self.path)
        if config.split == "held_out" and any(
            record.get("config", {}).get("split") == "held_out" for record in records
        ):
            raise HeldOutReuse("held_out may only be recorded once")

        if config.split == "dev":
            tuning_runs = sum(
                record.get("config", {}).get("split") == "dev" for record in records
            )
            if tuning_runs >= 3:
                raise TuningCapExceeded("at most three dev tuning runs are allowed")
            leaked = set(metadata.get("question_ids", [])) & self.held_out_ids
            if leaked:
                raise ValueError("held-out question IDs cannot appear in dev tuning runs")

        if config.parent_run_id:
            parent = next(
                (
                    record
                    for record in records
                    if record.get("run_id") == config.parent_run_id
                ),
                None,
            )
            if parent is None:
                raise ValueError("parent run ID was not found")
            parent_config = RunConfig(**parent["config"])
            changes = diff_fields(parent_config, config)
            if len(changes) > 1:
                raise MultiVariableChange(
                    f"run changes multiple tunable fields: {', '.join(sorted(changes))}"
                )

        run_id = uuid.uuid4().hex
        record = {
            "run_id": run_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config),
            "metadata": build_metadata(metadata),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return run_id


def build_metadata(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if key in _METADATA_ALLOWLIST}


def _serialize_for_secret_scan(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def assert_no_secrets(obj: Any, forbidden: list[str]) -> None:
    serialized = _serialize_for_secret_scan(obj)
    for secret in forbidden:
        if secret and re.search(re.escape(secret), serialized, flags=re.IGNORECASE):
            raise ValueError("secret value found in evaluation metadata or results")
