from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass(frozen=True, slots=True)
class SourceConfig:
    name: str
    domain: str
    base_url: str
    rss_urls: tuple[str, ...] = field(default_factory=tuple)
    section_urls: tuple[str, ...] = field(default_factory=tuple)
    parser_type: str = ""
    article_url_patterns: tuple[str, ...] = field(default_factory=tuple)
    archive_url_template: str | None = None


SOURCE_CONFIGS: dict[str, SourceConfig] = {
    "lenta": SourceConfig(
        name="Lenta.ru",
        domain="lenta.ru",
        base_url="https://lenta.ru",
        rss_urls=(
            "https://lenta.ru/rss",
            "https://lenta.ru/rss/news",
        ),
        section_urls=(
            "https://lenta.ru/news/",
            "https://lenta.ru/rubrics/russia/",
        ),
        parser_type="lenta",
        article_url_patterns=(
            r"^https?://lenta\.ru/news/\d{4}/\d{2}/\d{2}/[^/]+/?$",
            r"^https?://lenta\.ru/articles/\d{4}/\d{2}/\d{2}/[^/]+/?$",
        ),
        archive_url_template="https://lenta.ru/{year}/{month}/{day}/",
    ),
    "ria": SourceConfig(
        name="РИА Новости",
        domain="ria.ru",
        base_url="https://ria.ru",
        rss_urls=(
            "https://ria.ru/export/rss2/index.xml",
            "https://ria.ru/export/rss2/archive/index.xml",
        ),
        section_urls=(
            "https://ria.ru/world/",
            "https://ria.ru/politics/",
        ),
        parser_type="ria",
        article_url_patterns=(
            r"^https?://ria\.ru/\d{8}/[^/]+\.html$",
            r"^https?://ria\.ru/\d{8}/[^/]+/?$",
        ),
        archive_url_template="https://ria.ru/{ymd}/",
    ),
}


def get_source_config(source_name: str) -> SourceConfig:
    try:
        return SOURCE_CONFIGS[source_name]
    except KeyError as exc:
        raise ValueError(f"Unknown source config: {source_name}") from exc
