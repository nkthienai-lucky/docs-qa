from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal


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
    generation_k: int = 5


_TUNABLE_FIELDS = (
    "chunk_size",
    "overlap",
    "top_k",
    "generation_k",
    "candidates",
    "rrf_k",
    "prompt_hash",
)
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
    "generation_k",
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
        field for field in _TUNABLE_FIELDS if getattr(a, field) != getattr(b, field)
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
                raise ValueError(
                    f"invalid run log JSON on line {line_number}"
                ) from error
            if not isinstance(record, dict):
                raise TypeError(f"run log line {line_number} must be an object")
            records.append(record)
    return records


class RunLog:
    def __init__(self, path: Path, held_out_ids: set[str] | None = None) -> None:
        self.path = path
        self.held_out_ids = held_out_ids or set()

    def held_out_config_names(self) -> set[str]:
        return {
            record.get("config", {}).get("name")
            for record in _read_records(self.path)
            if record.get("config", {}).get("split") == "held_out"
            and isinstance(record.get("config", {}).get("name"), str)
        }

    def record(self, config: RunConfig, metadata: dict[str, Any]) -> str:
        return self.record_batch([(config, metadata)])[0]

    def record_batch(
        self, entries: list[tuple[RunConfig, dict[str, Any]]]
    ) -> list[str]:
        if not entries:
            return []
        records = _read_records(self.path)
        pending: list[dict[str, Any]] = []
        run_ids: list[str] = []
        for config, metadata in entries:
            prior_configs = [record.get("config", {}) for record in records + pending]

            if config.split == "dev":
                if config.tuning_round < 0:
                    raise ValueError("dev tuning_round cannot be negative")
                rounds = {
                    item.get("tuning_round", 0)
                    for item in prior_configs
                    if item.get("split") == "dev"
                }
                if config.tuning_round not in rounds and len(rounds) >= 3:
                    raise TuningCapExceeded("at most three dev tuning rounds are allowed")
                next_round = max(rounds, default=-1) + 1
                if config.tuning_round > next_round:
                    raise ValueError("dev tuning rounds cannot be skipped")
                if any(
                    item.get("split") == "dev"
                    and item.get("name") == config.name
                    and item.get("tuning_round", 0) == config.tuning_round
                    for item in prior_configs
                ):
                    raise ValueError(
                        "configuration already recorded in this dev tuning round"
                    )
                leaked = set(metadata.get("question_ids", [])) & self.held_out_ids
                if leaked:
                    raise ValueError(
                        "held-out question IDs cannot appear in dev tuning runs"
                    )
            elif config.split == "held_out":
                if config.tuning_round != 0:
                    raise ValueError("held-out runs cannot have a tuning round")
                if any(
                    item.get("split") == "held_out" and item.get("name") == config.name
                    for item in prior_configs
                ):
                    raise HeldOutReuse(
                        f"held_out configuration {config.name!r} may only be recorded once"
                    )
                question_ids = set(metadata.get("question_ids", []))
                if self.held_out_ids and question_ids != self.held_out_ids:
                    raise ValueError(
                        "held-out runs must use the complete frozen held-out split"
                    )
            else:
                raise ValueError(f"unsupported split: {config.split}")

            if config.parent_run_id:
                parent = next(
                    (
                        record
                        for record in records + pending
                        if record.get("run_id") == config.parent_run_id
                    ),
                    None,
                )
                if parent is None:
                    raise ValueError("parent run ID was not found")
                parent_config = RunConfig(**parent["config"])
                if parent_config.split != config.split:
                    raise ValueError("parent run must use the same split")
                changes = diff_fields(parent_config, config)
                if len(changes) > 1:
                    raise MultiVariableChange(
                        f"run changes multiple tunable fields: {', '.join(sorted(changes))}"
                    )

            run_id = uuid.uuid4().hex
            run_ids.append(run_id)
            pending.append(
                {
                    "run_id": run_id,
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "config": asdict(config),
                    "metadata": build_metadata(metadata),
                }
            )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.writelines(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in pending
            )
        return run_ids


def build_metadata(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if key in _METADATA_ALLOWLIST}


def _serialize_for_secret_scan(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def assert_no_secrets(obj: Any, forbidden: list[str]) -> None:
    serialized = _serialize_for_secret_scan(obj)
    for secret in forbidden:
        if secret and re.search(re.escape(secret), serialized, flags=re.IGNORECASE):
            raise ValueError("secret value found in evaluation metadata or results")
