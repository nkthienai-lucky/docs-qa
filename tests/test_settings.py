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


def test_generation_limits_default_to_efficient_positive_values() -> None:
    settings = Settings(_env_file=None)
    assert settings.num_ctx == 4096
    assert settings.generator_num_predict == 256
    assert settings.reranker_device == "cpu"
    assert settings.generation_k == 1


def test_generation_limits_must_be_positive() -> None:
    with pytest.raises(ValueError, match="num_ctx and generator_num_predict"):
        Settings(_env_file=None, num_ctx=0)
    with pytest.raises(ValueError, match="num_ctx and generator_num_predict"):
        Settings(_env_file=None, generator_num_predict=0)


def test_reranker_settings_default_and_validate() -> None:
    settings = Settings(_env_file=None)
    assert settings.reranker_device == "cpu"
    assert settings.reranker_batch_size is None
    assert settings.generation_k == 1
    with pytest.raises(ValueError, match="generation_k"):
        Settings(_env_file=None, generation_k=0)
    with pytest.raises(ValueError, match="generation_k"):
        Settings(_env_file=None, generation_k=6)
    with pytest.raises(ValueError, match="reranker_batch_size"):
        Settings(_env_file=None, reranker_batch_size=0)
    with pytest.raises(ValueError, match="reranker_device"):
        Settings(_env_file=None, reranker_device="bad-device")


def test_env_example_contains_no_secret_values() -> None:
    example = (ROOT / ".env.example").read_text()
    assert "sk-" not in example
    assert "ANTHROPIC_AUTH_TOKEN=" in example
    assert "sk-" not in example
