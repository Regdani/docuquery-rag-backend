"""Application settings loaded from environment variables and an optional `.env` file."""

from functools import lru_cache
from typing import Any

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration: providers, model names, URLs, timeouts and limits."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_ignore_empty=True,
    )

    # Provider selection
    llm_provider: str = "ollama"
    embedding_provider: str | None = Field(
        default=None, description="Defaults to llm_provider when unset."
    )

    # Azure OpenAI
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str = ""
    azure_openai_embedding_deployment: str = ""

    # Google Gemini
    google_api_key: str = ""
    gemini_chat_model: str = "gemini-3.1-flash-lite"
    gemini_embedding_model: str = "gemini-embedding-001"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.2:3b"
    ollama_embedding_model: str = "nomic-embed-text"

    # Vector store and data
    index_dir: str = "./data/index"
    data_dir: str = "./data/sample"
    chroma_collection: str = "manuals"

    # Chunking and retrieval
    chunk_size: int = 800
    chunk_overlap: int = 100
    retrieval_top_k: int = 4

    # Timeouts, retries and limits
    llm_timeout_seconds: float = 60.0
    embedding_timeout_seconds: float = 30.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5
    max_question_length: int = 1000
    max_concurrent_requests: int = 8

    # Logging
    log_level: str = "INFO"

    @field_validator("llm_provider", "embedding_provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: Any, info: ValidationInfo) -> Any:
        # Unknown values are rejected by the provider factory with ProviderConfigError.
        if isinstance(value, str):
            value = value.strip().lower()
            if not value:
                return None if info.field_name == "embedding_provider" else "ollama"
        return value

    @model_validator(mode="after")
    def _default_embedding_provider(self) -> "Settings":
        if self.embedding_provider is None:
            self.embedding_provider = self.llm_provider
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
