"""Application settings loaded from environment variables and an optional `.env` file."""

from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, field_validator
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

    # Provider selection: all three are required; the factory raises ProviderConfigError
    # naming the variable when one is missing. There are no defaults or fallbacks.
    llm_provider: str | None = Field(default=None, description="gemini | ollama")
    embedding_provider: str | None = Field(default=None, description="gemini | ollama")
    embedding_model: str = Field(default="", description="Embedding model of EMBEDDING_PROVIDER.")

    # Google Gemini
    google_api_key: str = ""
    gemini_chat_model: str = "gemini-3.1-flash-lite"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.2:3b"

    # Vector store and data
    index_dir: str = "./data/index"
    data_dir: str = "./data/sample"
    chroma_collection: str = "manuals"

    # Chunking and retrieval
    chunk_size: int = 800
    chunk_overlap: int = 100
    retrieval_top_k: int = Field(
        default=4, ge=1, le=20, description="Number of chunks retrieved per question (1..20)."
    )
    retrieval_relevance_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score (0..1) for a retrieved chunk to be used.",
    )

    snippet_max_chars: int = Field(
        default=200, ge=20, description="Maximum length of the source snippet in answers."
    )

    # Timeouts, retries and limits
    llm_timeout_seconds: float = 60.0
    embedding_timeout_seconds: float = 30.0
    health_llm_timeout_seconds: float = Field(
        default=5.0, gt=0, description="Timeout of the optional LLM check in /health/ready."
    )
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5
    max_question_length: int = Field(default=1000, ge=1)
    rate_limit_retry_after_seconds: int = Field(
        default=5,
        ge=1,
        description="Retry-After value sent with 503 LLM_RATE_LIMITED when the provider gave none.",
    )
    max_concurrent_requests: int = 8
    history_max_messages: int = Field(
        default=20, ge=1, description="Most recent messages kept per session."
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_question_max_chars: int = Field(default=80, ge=0)

    @field_validator("llm_provider", "embedding_provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: Any) -> Any:
        # Missing and unknown values are rejected by the provider factory (ProviderConfigError).
        if isinstance(value, str):
            return value.strip().lower() or None
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
