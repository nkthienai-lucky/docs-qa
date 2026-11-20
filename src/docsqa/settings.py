from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeProfile(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"


class Settings(BaseSettings):
    runtime_profile: RuntimeProfile = RuntimeProfile.LOCAL
    runtime_generation_provider: Literal["ollama", "openai_compatible"] = "ollama"
    runtime_embedding_provider: Literal["ollama", "openai_compatible"] = "ollama"
    runtime_rerank_provider: Literal["local", "api"] = "local"
    runtime_chat_base_url: SecretStr | None = None
    runtime_chat_token: SecretStr | None = None
    runtime_chat_model: str | None = None
    runtime_embedding_base_url: SecretStr | None = None
    runtime_embedding_token: SecretStr | None = None
    runtime_embedding_model: str | None = None
    runtime_rerank_base_url: SecretStr | None = None
    runtime_rerank_token: SecretStr | None = None
    runtime_rerank_model: str | None = None
    runtime_embedding_query_instruction: str = (
        "Given a web search query, retrieve relevant passages"
    )
    runtime_api_timeout_s: float = Field(default=30, gt=0, le=120, allow_inf_nan=False)
    runtime_upload_dir: str = "data/uploads"
    runtime_deployment_approved: bool = False
    database_url: str = "postgresql://roadmap:roadmap@localhost:5432/docsqa"
    ollama_host: str = "http://127.0.0.1:11434"
    generator_model: str = "qwen3.5:9b"
    embed_model: str = "qwen3-embedding:0.6b"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"
    reranker_batch_size: int | None = None
    generation_k: int = 1
    # 4096 avoids wasting scarce VRAM on a context window far larger than the
    # top-five retrieved passages used by this app.
    num_ctx: int = 4096
    generator_num_predict: int = 256

    def model_post_init(self, __context, /) -> None:
        if self.num_ctx <= 0 or self.generator_num_predict <= 0:
            raise ValueError("num_ctx and generator_num_predict must be positive")
        if self.reranker_batch_size is not None and self.reranker_batch_size <= 0:
            raise ValueError("reranker_batch_size must be positive")
        if self.reranker_device not in {"auto", "cpu", "cuda", "cuda:0"}:
            raise ValueError("reranker_device must be auto, cpu, cuda, or cuda:0")
        if self.generation_k <= 0 or self.generation_k > 5:
            raise ValueError("generation_k must be between 1 and 5")

    api_judge_base_url: SecretStr | None = None
    api_judge_token: SecretStr | None = None
    api_judge_model: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        repr=False,
        hide_input_in_errors=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
