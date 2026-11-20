from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .settings import RuntimeProfile, Settings

if TYPE_CHECKING:
    from .embedding import Embedder
    from .store import Hit


class RuntimeConfigurationError(ValueError):
    """Invalid runtime configuration, with secret-free diagnostics."""


class Generator(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str: ...


class RuntimeReranker(Protocol):
    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]: ...


@dataclass(frozen=True)
class RuntimeComponents:
    generator: Generator
    embedder: Embedder
    reranker: RuntimeReranker


def _valid_url(value: str, cloud: bool) -> bool:
    from .provider_http import ProviderHTTPError, provider_endpoint

    try:
        normalized = provider_endpoint(value, "embeddings")
        return not cloud or normalized.startswith("https://")
    except ProviderHTTPError:
        return False


def validate_runtime_profile(settings: Settings) -> None:
    cloud = settings.runtime_profile == RuntimeProfile.CLOUD
    selectors = (
        settings.runtime_generation_provider,
        settings.runtime_embedding_provider,
        settings.runtime_rerank_provider,
    )
    if cloud and selectors != ("openai_compatible", "openai_compatible", "api"):
        raise RuntimeConfigurationError("cloud requires three API providers")
    for side, selected in zip(("chat", "embedding", "rerank"), selectors):
        if selected not in {"openai_compatible", "api"}:
            continue
        for part in ("base_url", "token", "model"):
            value = getattr(settings, f"runtime_{side}_{part}")
            value = (
                value.get_secret_value()
                if hasattr(value, "get_secret_value")
                else value
            )
            if (
                not value
                or not value.strip()
                or value != value.strip()
                or any(ord(c) < 32 or ord(c) == 127 for c in value)
            ):
                raise RuntimeConfigurationError(f"runtime_{side}_{part} is invalid")
            if part == "base_url" and not _valid_url(value, cloud):
                raise RuntimeConfigurationError(f"runtime_{side}_base_url is invalid")
    if cloud and not Path(settings.runtime_upload_dir).is_absolute():
        raise RuntimeConfigurationError("cloud upload path must be absolute")


def build_runtime_components(settings: Settings) -> RuntimeComponents:
    validate_runtime_profile(settings)
    from .embedding import OllamaEmbedder, OpenAICompatibleEmbedder
    from .generation import OllamaGenerator, OpenAICompatibleGenerator

    def api_kwargs(side):
        return {
            "host": getattr(settings, f"runtime_{side}_base_url").get_secret_value(),
            "token": getattr(settings, f"runtime_{side}_token").get_secret_value(),
            "model": getattr(settings, f"runtime_{side}_model"),
            "timeout": settings.runtime_api_timeout_s,
        }

    generator = (
        OpenAICompatibleGenerator(
            **api_kwargs("chat"), max_tokens=settings.generator_num_predict
        )
        if settings.runtime_generation_provider == "openai_compatible"
        else OllamaGenerator(
            host=settings.ollama_host,
            model=settings.generator_model,
            num_ctx=settings.num_ctx,
            num_predict=settings.generator_num_predict,
        )
    )
    embedder = (
        OpenAICompatibleEmbedder(
            **api_kwargs("embedding"),
            query_instruction=settings.runtime_embedding_query_instruction,
        )
        if settings.runtime_embedding_provider == "openai_compatible"
        else OllamaEmbedder(
            host=settings.ollama_host,
            model=settings.embed_model,
            query_instruction=settings.runtime_embedding_query_instruction,
        )
    )
    if settings.runtime_rerank_provider == "api":
        from .api_rerank import ApiReranker

        reranker = ApiReranker(**api_kwargs("rerank"))
    else:
        from .rerank import Reranker

        reranker = Reranker(
            settings.reranker_model,
            device=settings.reranker_device,
            batch_size=settings.reranker_batch_size,
        )
    return RuntimeComponents(generator, embedder, reranker)


def runtime_index_identity(settings: Settings, embedder: Embedder) -> str:
    import hashlib
    import json
    from urllib.parse import urlsplit, urlunsplit

    from .provider_http import provider_endpoint

    if type(embedder.dim) is not int or not 1 <= embedder.dim <= 2000:
        raise RuntimeConfigurationError(
            "embedding dimension requires a compatible fresh index"
        )
    if settings.runtime_embedding_provider == "openai_compatible":
        host = settings.runtime_embedding_base_url
        host = host.get_secret_value() if hasattr(host, "get_secret_value") else host
        endpoint = provider_endpoint(host, "embeddings")
    else:
        if not _valid_url(settings.ollama_host, False):
            raise RuntimeConfigurationError("invalid local embedding endpoint")
        parsed = urlsplit(settings.ollama_host)
        # Preserve any path prefix (reverse proxies can route by path).
        # OllamaEmbedder endpoints are exactly <host>/api/embed.
        base = parsed.path.rstrip("/")
        endpoint = urlunsplit(
            (parsed.scheme, parsed.netloc.lower(), f"{base}/api/embed", "", "")
        )
    data = {
        "provider": embedder.backend,
        "endpoint": endpoint,
        "model": embedder.model,
        "dimension": embedder.dim,
        "query_instruction": embedder.query_instruction,
        "preprocessing": "canonical-text-v1",
    }
    return (
        "runtime-v1:"
        + hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
