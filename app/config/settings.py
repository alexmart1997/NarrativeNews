from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


DEFAULT_DB_PATH = Path("data") / "narrative_news.db"


@dataclass(frozen=True, slots=True)
class Settings:
    database_path: Path
    log_level: str = "INFO"
    log_format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    llm_provider: str = "local_llama"
    llm_host: str = "127.0.0.1"
    llm_port: int = 11434
    llm_base_url: str = "http://127.0.0.1:11434"
    llm_model_name: str = "qwen2.5:7b"
    llm_timeout_seconds: float = 30.0
    llm_temperature: float = 0.2
    llm_max_tokens: int = 512


@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    database_path = Path(os.getenv("NARRATIVE_NEWS_DB_PATH", DEFAULT_DB_PATH))
    log_level = os.getenv("NARRATIVE_NEWS_LOG_LEVEL", "INFO")
    llm_provider = os.getenv("NARRATIVE_NEWS_LLM_PROVIDER", "local_llama")
    llm_host = os.getenv("NARRATIVE_NEWS_LLM_HOST", "127.0.0.1")
    llm_port = int(os.getenv("NARRATIVE_NEWS_LLM_PORT", "11434"))
    llm_base_url = os.getenv("NARRATIVE_NEWS_LLM_BASE_URL", f"http://{llm_host}:{llm_port}")
    llm_model_name = os.getenv("NARRATIVE_NEWS_LLM_MODEL_NAME", "qwen2.5:7b")
    llm_timeout_seconds = float(os.getenv("NARRATIVE_NEWS_LLM_TIMEOUT", "30"))
    llm_temperature = float(os.getenv("NARRATIVE_NEWS_LLM_TEMPERATURE", "0.2"))
    llm_max_tokens = int(os.getenv("NARRATIVE_NEWS_LLM_MAX_TOKENS", "512"))
    return Settings(
        database_path=database_path,
        log_level=log_level,
        llm_provider=llm_provider,
        llm_host=llm_host,
        llm_port=llm_port,
        llm_base_url=llm_base_url,
        llm_model_name=llm_model_name,
        llm_timeout_seconds=llm_timeout_seconds,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
    )


def get_settings(database_path: Path | None = None) -> Settings:
    base_settings = _cached_settings()
    if database_path is None:
        return base_settings
    return Settings(
        database_path=database_path,
        log_level=base_settings.log_level,
        log_format=base_settings.log_format,
        llm_provider=base_settings.llm_provider,
        llm_host=base_settings.llm_host,
        llm_port=base_settings.llm_port,
        llm_base_url=base_settings.llm_base_url,
        llm_model_name=base_settings.llm_model_name,
        llm_timeout_seconds=base_settings.llm_timeout_seconds,
        llm_temperature=base_settings.llm_temperature,
        llm_max_tokens=base_settings.llm_max_tokens,
    )
