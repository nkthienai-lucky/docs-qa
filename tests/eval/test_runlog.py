from __future__ import annotations

import json
from pathlib import Path

import pytest

from docsqa.eval.runlog import (
    HeldOutReuse,
    MultiVariableChange,
    RunConfig,
    RunLog,
    TuningCapExceeded,
    assert_no_secrets,
    build_metadata,
    diff_fields,
)


def config(name: str, **updates) -> RunConfig:
    values = {
        "name": name,
        "parent_run_id": None,
        "chunk_size": 512,
        "overlap": 64,
        "top_k": 5,
        "candidates": 50,
        "rrf_k": 60,
        "prompt_hash": "prompt-v1",
        "split": "dev",
    }
    values.update(updates)
    return RunConfig(**values)


def test_diff_fields_and_multi_variable_run_rejected(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    log = RunLog(path)
    first_id = log.record(config("base"), {})
    changed = config(
        "invalid",
        parent_run_id=first_id,
        chunk_size=256,
        top_k=10,
    )

    assert diff_fields(config("base"), changed) == {"chunk_size", "top_k"}
    with pytest.raises(MultiVariableChange):
        log.record(changed, {})


def test_held_out_recorded_once_only(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "runs.jsonl")
    held_out = config("final", split="held_out")
    log.record(held_out, {})

    with pytest.raises(HeldOutReuse):
        log.record(config("final-repeat", split="held_out"), {})


def test_tuning_cap_is_three(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "runs.jsonl")
    for index in range(3):
        log.record(config(f"dev-{index}"), {})

    with pytest.raises(TuningCapExceeded):
        log.record(config("dev-four"), {})


def test_metadata_allow_list_drops_base_url_headers_and_env() -> None:
    metadata = build_metadata(
        {"git_revision": "abc", "model": "qwen", "base_url": "https://private", "headers": {"Authorization": "secret"}, "env": {"TOKEN": "secret"}, "unknown": "drop"}
    )

    assert metadata == {"git_revision": "abc", "model": "qwen"}
    assert "base_url" not in metadata and "headers" not in metadata and "env" not in metadata


def test_assert_no_secrets_detects_token_and_gateway_host() -> None:
    with pytest.raises(ValueError, match="secret"):
        assert_no_secrets({"result": "token is abc123"}, ["abc123"])
    with pytest.raises(ValueError, match="secret"):
        assert_no_secrets({"result": "gateway.private.test"}, ["gateway.private.test"])


def test_heldout_question_ids_never_in_tuning_runs(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    log = RunLog(path, held_out_ids={"q-held-1", "q-held-2"})
    log.record(config("dev"), {"question_ids": ["q-dev-1"]})
    record = json.loads(path.read_text(encoding="utf-8"))
    assert set(record["metadata"]["question_ids"]).isdisjoint({"q-held-1", "q-held-2"})

    with pytest.raises(ValueError, match="held-out"):
        log.record(config("dev-bad"), {"question_ids": ["q-held-1"]})
