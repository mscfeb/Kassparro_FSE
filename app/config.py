from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_dir: str = "data"
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    gemini_api_key: str | None = None
    database_url: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    provider_timeout_seconds: float = 15.0
    max_provider_concurrency: int = 3
    max_model_turns_per_answer: int = 3
    max_retries_per_turn: int = 1
    retry_base_delay_seconds: float = 1.0
    default_agent_version: str = "v1"
    development_split: str = "development"
    check_split: str = "check"

    def __post_init__(self) -> None:
        if self.provider != "gemini":
            raise ValueError("LLM_PROVIDER must be gemini")
        if self.provider_timeout_seconds != 15:
            raise ValueError("PROVIDER_TIMEOUT_SECONDS must remain 15")
        if not 1 <= self.max_provider_concurrency <= 3:
            raise ValueError("MAX_PROVIDER_CONCURRENCY must be between one and three")
        if not 1 <= self.max_model_turns_per_answer <= 3:
            raise ValueError("MAX_MODEL_TURNS_PER_ANSWER must be between one and three")
        if not 0 <= self.max_retries_per_turn <= 1:
            raise ValueError("MAX_RETRIES_PER_TURN must be zero or one")
        if self.retry_base_delay_seconds < 0:
            raise ValueError("RETRY_BASE_DELAY_SECONDS must not be negative")
        if self.default_agent_version not in {"v1", "v2"}:
            raise ValueError("DEFAULT_AGENT_VERSION must be v1 or v2")

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            data_dir=os.getenv("DATA_DIR", "data"),
            provider=os.getenv("LLM_PROVIDER", "gemini").lower(),
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
            database_url=os.getenv("DATABASE_URL"),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            langfuse_host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            provider_timeout_seconds=float(os.getenv("PROVIDER_TIMEOUT_SECONDS", "15")),
            max_provider_concurrency=int(os.getenv("MAX_PROVIDER_CONCURRENCY", "3")),
            max_model_turns_per_answer=int(os.getenv("MAX_MODEL_TURNS_PER_ANSWER", "3")),
            max_retries_per_turn=int(os.getenv("MAX_RETRIES_PER_TURN", "1")),
            retry_base_delay_seconds=float(os.getenv("RETRY_BASE_DELAY_SECONDS", "1")),
            default_agent_version=os.getenv("DEFAULT_AGENT_VERSION", "v1"),
            development_split=os.getenv("DEVELOPMENT_SPLIT", "development"),
            check_split=os.getenv("CHECK_SPLIT", "check"),
        )

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)