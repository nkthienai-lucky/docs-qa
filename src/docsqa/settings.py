from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql://roadmap:roadmap@localhost:5432/docsqa"
    ollama_host: str = "http://127.0.0.1:11434"
    generator_model: str = "qwen3.5:9b"
    embed_model: str = "qwen3-embedding:0.6b"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    num_ctx: int = 8192
    api_judge_base_url: SecretStr | None = None
    api_judge_token: SecretStr | None = None
    api_judge_model: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        repr=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
