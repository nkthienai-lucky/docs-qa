import itertools
import os
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest

from docsqa.settings import Settings


@pytest.fixture(autouse=True)
def clean_runtime_environment(monkeypatch):
    for key in os.environ:
        if key.startswith(("RUNTIME_", "API_JUDGE_")):
            monkeypatch.delenv(key)


def cloud_values():
    values = {
        "runtime_profile": "cloud",
        "runtime_generation_provider": "openai_compatible",
        "runtime_embedding_provider": "openai_compatible",
        "runtime_rerank_provider": "api",
        "runtime_upload_dir": "/var/data/uploads",
    }
    for side in ("chat", "embedding", "rerank"):
        values.update(
            {
                f"runtime_{side}_base_url": "https://provider.invalid/v1",
                f"runtime_{side}_token": "sentinel-token",
                f"runtime_{side}_model": "test-model",
            }
        )
    return values


def validate(values):
    from docsqa.runtime import validate_runtime_profile

    settings = Settings(_env_file=None, **values)
    validate_runtime_profile(settings)
    return settings


def test_runtime_defaults_are_local():
    settings = validate({})
    assert settings.runtime_profile == "local"
    assert settings.runtime_generation_provider == "ollama"
    assert settings.runtime_embedding_provider == "ollama"
    assert settings.runtime_rerank_provider == "local"
    assert settings.runtime_api_timeout_s == 30
    assert Path(settings.runtime_upload_dir) == Path("data/uploads")
    assert settings.runtime_deployment_approved is False
    assert settings.generator_model == "qwen3.5:9b"


@pytest.mark.parametrize(
    "generation,embedding,rerank",
    itertools.product(
        ("ollama", "openai_compatible"),
        ("ollama", "openai_compatible"),
        ("local", "api"),
    ),
)
def test_local_provider_choices_are_independent(generation, embedding, rerank):
    values = cloud_values() | {
        "runtime_profile": "local",
        "runtime_generation_provider": generation,
        "runtime_embedding_provider": embedding,
        "runtime_rerank_provider": rerank,
    }
    validate(values)


@pytest.mark.parametrize(
    "field,value",
    [
        ("runtime_generation_provider", "ollama"),
        ("runtime_embedding_provider", "ollama"),
        ("runtime_rerank_provider", "local"),
    ],
)
def test_cloud_requires_three_explicit_api_providers(field, value):
    with pytest.raises(ValueError):
        validate(cloud_values() | {field: value})


@pytest.mark.parametrize("side", ["chat", "embedding", "rerank"])
@pytest.mark.parametrize("part", ["base_url", "model", "token"])
@pytest.mark.parametrize("value", [None, "", " ", "bad\nvalue"])
def test_cloud_rejects_each_incomplete_api(side, part, value):
    with pytest.raises(ValueError):
        validate(cloud_values() | {f"runtime_{side}_{part}": value})


def test_runtime_never_uses_judge_credentials(monkeypatch):
    for key in ("BASE_URL", "TOKEN", "MODEL"):
        monkeypatch.setenv(f"API_JUDGE_{key}", "judge-only")
    values = cloud_values()
    del values["runtime_chat_token"]
    with pytest.raises(ValueError):
        validate(values)


def test_cloud_ignores_unused_ollama_variables(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://unused.invalid")
    validate(cloud_values())


@pytest.mark.parametrize(
    "url",
    [
        "http://provider.invalid",
        "https://u:p@provider.invalid",
        "https://provider.invalid?q=secret",
        "https://provider.invalid/#secret",
        "https:///bad",
        "https://provider.invalid:bad",
        "https://provider.invalid/has space",
    ],
)
def test_cloud_rejects_unsafe_urls(url):
    with pytest.raises(ValueError):
        validate(cloud_values() | {"runtime_chat_base_url": url})


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), 121])
def test_runtime_timeout_is_bounded(timeout):
    with pytest.raises(ValueError):
        validate({"runtime_api_timeout_s": timeout})


def test_cloud_upload_path_must_be_absolute():
    with pytest.raises(ValueError):
        validate(cloud_values() | {"runtime_upload_dir": "data/uploads"})


def test_runtime_errors_hide_input_values():
    for values in (
        {"runtime_profile": "sentinel-secret"},
        cloud_values() | {"runtime_chat_base_url": "https://sentinel-secret@bad"},
    ):
        with pytest.raises(ValueError) as caught:
            validate(values)
        assert "sentinel-secret" not in "".join(
            traceback.format_exception(caught.value)
        )


def test_runtime_secrets_masked():
    settings = validate(cloud_values())
    assert "sentinel-token" not in repr(settings)
    assert "sentinel-token" not in settings.model_dump_json()
    assert "provider.invalid" not in settings.model_dump_json()


def test_component_factory_api_only_no_network(monkeypatch):
    from docsqa.runtime import RuntimeConfigurationError, build_runtime_components

    constructed = []

    def constructor(name):
        def build(**kwargs):
            constructed.append((name, kwargs))
            return name

        return build

    monkeypatch.setattr(
        "docsqa.generation.OpenAICompatibleGenerator", constructor("chat")
    )
    monkeypatch.setattr(
        "docsqa.embedding.OpenAICompatibleEmbedder", constructor("embed")
    )
    monkeypatch.setattr("docsqa.api_rerank.ApiReranker", constructor("rank"))

    def forbidden(*args, **kwargs):
        pytest.fail("local provider or HTTP constructed")

    monkeypatch.setattr("docsqa.generation.OllamaGenerator", forbidden)
    monkeypatch.setattr("docsqa.embedding.OllamaEmbedder", forbidden)
    monkeypatch.setattr("docsqa.rerank.Reranker", forbidden)
    monkeypatch.setattr("docsqa.provider_http.post_json", forbidden)
    values = cloud_values()
    components = build_runtime_components(Settings(_env_file=None, **values))
    assert (components.generator, components.embedder, components.reranker) == (
        "chat",
        "embed",
        "rank",
    )
    assert len(constructed) == 3
    with pytest.raises(RuntimeConfigurationError):
        build_runtime_components(
            Settings(_env_file=None, **(values | {"runtime_rerank_provider": "local"}))
        )
    assert len(constructed) == 3


@pytest.mark.parametrize(
    "generation,embedding,rerank",
    itertools.product(
        ["ollama", "openai_compatible"],
        ["ollama", "openai_compatible"],
        ["local", "api"],
    ),
)
def test_factory_local_mixes(monkeypatch, generation, embedding, rerank):
    from docsqa.runtime import build_runtime_components

    for path, label in [
        ("generation.OllamaGenerator", "ollama"),
        ("generation.OpenAICompatibleGenerator", "openai_compatible"),
        ("embedding.OllamaEmbedder", "ollama"),
        ("embedding.OpenAICompatibleEmbedder", "openai_compatible"),
        ("rerank.Reranker", "local"),
        ("api_rerank.ApiReranker", "api"),
    ]:
        monkeypatch.setattr(
            "docsqa." + path, lambda *args, _label=label, **kwargs: _label
        )
    components = build_runtime_components(
        Settings(
            _env_file=None,
            **(
                cloud_values()
                | {
                    "runtime_profile": "local",
                    "runtime_generation_provider": generation,
                    "runtime_embedding_provider": embedding,
                    "runtime_rerank_provider": rerank,
                }
            ),
        )
    )
    assert (components.generator, components.embedder, components.reranker) == (
        generation,
        embedding,
        rerank,
    )


def test_identity_changes_when_local_path_prefix_changes():
    from docsqa.runtime import runtime_index_identity

    embedder = SimpleNamespace(
        model="m", backend="ollama", dim=2, query_instruction="q"
    )
    base = Settings(_env_file=None, ollama_host="http://host.invalid")
    a = runtime_index_identity(base, embedder)
    changed = Settings(_env_file=None, ollama_host="http://host.invalid/team-b")
    b = runtime_index_identity(changed, embedder)
    assert a != b
    # Trailing slash is equivalent to no trailing slash.
    slash = Settings(_env_file=None, ollama_host="http://host.invalid/team-b/")
    assert b == runtime_index_identity(slash, embedder)
