from __future__ import annotations

from datetime import date, timedelta
import logging
import re
from urllib.parse import urljoin, urlparse

from app.ingestion.fetcher import FetchError, HttpFetcher
from app.ingestion.sources import SourceConfig

logger = logging.getLogger(__name__)


class ArchiveDiscoveryService:
    ARTICLE_HREF_RE = re.compile(r"""href=["'](?P<url>[^"']+)["']""", re.IGNORECASE)

    def __init__(self, fetcher: HttpFetcher) -> None:
        self.fetcher = fetcher

    def discover_for_date_range(
        self,
        source_config: SourceConfig,
        *,
        date_from: date,
        date_to: date,
        per_day_limit: int | None = None,
    ) -> dict[date, list[str]]:
        if source_config.archive_url_template is None:
            return {}

        discovered: dict[date, list[str]] = {}
        current_date = date_from
        while current_date <= date_to:
            urls = self.discover_for_date(
                source_config,
                target_date=current_date,
                limit=per_day_limit,
            )
            discovered[current_date] = urls
            current_date += timedelta(days=1)
        return discovered

    def discover_for_date(
        self,
        source_config: SourceConfig,
        *,
        target_date: date,
        limit: int | None = None,
    ) -> list[str]:
        archive_url = self._build_archive_url(source_config, target_date)
        if archive_url is None:
            return []

        try:
            response = self.fetcher.fetch(archive_url)
        except FetchError:
            logger.exception("Failed to fetch archive page %s", archive_url)
            return []

        return self._extract_urls(
            response.text,
            base_url=archive_url,
            source_config=source_config,
            limit=limit,
        )

    @staticmethod
    def _build_archive_url(source_config: SourceConfig, target_date: date) -> str | None:
        if source_config.archive_url_template is None:
            return None
        return source_config.archive_url_template.format(
            year=f"{target_date.year:04d}",
            month=f"{target_date.month:02d}",
            day=f"{target_date.day:02d}",
            ymd=f"{target_date.year:04d}{target_date.month:02d}{target_date.day:02d}",
        )

    def _extract_urls(
        self,
        html: str,
        *,
        base_url: str,
        source_config: SourceConfig,
        limit: int | None,
    ) -> list[str]:
        urls: list[str] = []
        seen: set[str] = set()
        for href in self.ARTICLE_HREF_RE.findall(html):
            absolute = urljoin(base_url, href)
            if not self._is_article_url(absolute, source_config):
                continue
            if absolute in seen:
                continue
            seen.add(absolute)
            urls.append(absolute)
            if limit is not None and len(urls) >= limit:
                break
        return urls

    @staticmethod
    def _is_article_url(url: str, source_config: SourceConfig) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return False
        if source_config.domain not in parsed.netloc:
            return False
        if not parsed.path or parsed.path == "/":
            return False
        if source_config.article_url_patterns:
            return any(re.match(pattern, url) for pattern in source_config.article_url_patterns)
        return True
