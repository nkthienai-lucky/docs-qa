from pathlib import Path

import pytest

from docsqa.settings import Settings


ROOT = Path(__file__).parents[1]


def test_secret_fields_masked_in_repr_and_dump() -> None:
    settings = Settings(api_judge_token="sk-x")

    assert "sk-x" not in repr(settings)
    assert "sk-x" not in settings.model_dump_json()


def test_env_file_ignored_example_tracked() -> None:
    gitignore = (ROOT / ".gitignore").read_text()
    example = (ROOT / ".env.example").read_text()

    assert ".env" in gitignore
    assert "!.env.example" in gitignore
    assert "ANTHROPIC_AUTH_TOKEN=" in example
    assert "ANTHROPIC_BASE_URL=" in example
    assert "ANTHROPIC_DEFAULT_SONNET_MODEL=" in example
    assert "sk-" not in example
    return


def test_env_example_contains_no_secret_values() -> None:
    example = (ROOT / ".env.example").read_text()
    assert "sk-" not in example
    assert "ANTHROPIC_AUTH_TOKEN=" in example
    assert "sk-" not in example
