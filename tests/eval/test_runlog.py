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


def test_each_held_out_configuration_recorded_once(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "runs.jsonl")
    log.record(config("v1", split="held_out"), {})
    log.record(config("v2a", split="held_out"), {})

    with pytest.raises(HeldOutReuse):
        log.record(config("v1", split="held_out"), {})


def test_tuning_cap_counts_distinct_rounds_not_configurations(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "runs.jsonl")
    log.record(config("v1", tuning_round=0), {})
    log.record(config("v2a", tuning_round=0), {})
    log.record(config("v1", tuning_round=1), {})
    log.record(config("v1", tuning_round=2), {})

    with pytest.raises(TuningCapExceeded):
        log.record(config("v1", tuning_round=3), {})


def test_held_out_config_must_use_complete_frozen_split(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "runs.jsonl", held_out_ids={"q1", "q2"})
    with pytest.raises(ValueError, match="complete frozen"):
        log.record(
            config("v1", split="held_out"),
            {"question_ids": ["q1"], "expected_configs": ["v1", "v2a"]},
        )
    log.record(
        config("v1", split="held_out"),
        {"question_ids": ["q1", "q2"], "expected_configs": ["v1", "v2a"]},
    )
    log.record(
        config("v2a", split="held_out"),
        {"question_ids": ["q1", "q2"], "expected_configs": ["v1", "v2a"]},
    )
    with pytest.raises(HeldOutReuse):
        log.record(
            config("v2a", split="held_out"),
            {"question_ids": ["q1", "q2"], "expected_configs": ["v1", "v2a"]},
        )


def test_dev_configuration_cannot_repeat_within_round(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "runs.jsonl")
    log.record(config("v1", tuning_round=0), {})
    with pytest.raises(ValueError, match="already recorded"):
        log.record(config("v1", tuning_round=0), {})
    log.record(config("v2a", tuning_round=0), {})
    with pytest.raises(ValueError, match="skipped"):
        log.record(config("v1", tuning_round=2), {})


def test_tuning_cap_is_three(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "runs.jsonl")
    for round_number in range(3):
        log.record(config(f"dev-{round_number}", tuning_round=round_number), {})

    with pytest.raises(TuningCapExceeded):
        log.record(config("dev-four", tuning_round=3), {})


def test_held_out_recorded_once_only(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "runs.jsonl")
    held_out = config("final", split="held_out")
    log.record(held_out, {})
    with pytest.raises(HeldOutReuse):
        log.record(config("final", split="held_out"), {})


def test_tuning_cap_is_three_compatibility(tmp_path: Path) -> None:
    # Three independent configurations in three distinct rounds hit the cap.
    test_tuning_cap_is_three(tmp_path)


def test_metadata_allow_list_drops_base_url_headers_and_env() -> None:
    metadata = build_metadata(
        {
            "git_revision": "abc",
            "model": "qwen",
            "base_url": "https://private",
            "headers": {"Authorization": "secret"},
            "env": {"TOKEN": "secret"},
            "unknown": "drop",
        }
    )

    assert metadata == {"git_revision": "abc", "model": "qwen"}
    assert (
        "base_url" not in metadata
        and "headers" not in metadata
        and "env" not in metadata
    )


def test_assert_no_secrets_detects_token_and_gateway_host() -> None:
    with pytest.raises(ValueError, match="secret"):
        assert_no_secrets({"result": "token is abc123"}, ["abc123"])
    with pytest.raises(ValueError, match="secret"):
        assert_no_secrets({"result": "gateway.private.test"}, ["gateway.private.test"])




def test_dev_run_cannot_parent_from_held_out_run(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "runs.jsonl")
    held_id = log.record(config("held", split="held_out"), {})

    with pytest.raises(ValueError, match="same split"):
        log.record(
            config("dev-child", parent_run_id=held_id, chunk_size=256),
            {},
        )


def test_heldout_question_ids_never_in_tuning_runs(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    log = RunLog(path, held_out_ids={"q-held-1", "q-held-2"})
    log.record(config("dev"), {"question_ids": ["q-dev-1"]})
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert set(record["metadata"]["question_ids"]).isdisjoint({"q-held-1", "q-held-2"})

    with pytest.raises(ValueError, match="held-out"):
        log.record(config("dev-bad"), {"question_ids": ["q-held-1"]})
