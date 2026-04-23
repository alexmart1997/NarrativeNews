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


@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    database_path = Path(os.getenv("NARRATIVE_NEWS_DB_PATH", DEFAULT_DB_PATH))
    log_level = os.getenv("NARRATIVE_NEWS_LOG_LEVEL", "INFO")
    return Settings(database_path=database_path, log_level=log_level)


def get_settings(database_path: Path | None = None) -> Settings:
    base_settings = _cached_settings()
    if database_path is None:
        return base_settings
    return Settings(
        database_path=database_path,
        log_level=base_settings.log_level,
        log_format=base_settings.log_format,
    )
