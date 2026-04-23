from __future__ import annotations

import logging
import re
from urllib.parse import urljoin, urlparse

from app.ingestion.fetcher import FetchError, HttpFetcher
from app.ingestion.sources import SourceConfig

logger = logging.getLogger(__name__)


class SectionPageDiscoveryService:
    ARTICLE_HREF_RE = re.compile(r"""href=["'](?P<url>[^"']+)["']""", re.IGNORECASE)

    def __init__(self, fetcher: HttpFetcher) -> None:
        self.fetcher = fetcher

    def discover(self, source_config: SourceConfig, limit: int | None = None) -> list[str]:
        urls: list[str] = []
        seen: set[str] = set()

        for section_url in source_config.section_urls:
            try:
                response = self.fetcher.fetch(section_url)
            except FetchError:
                logger.exception("Failed to fetch section page %s", section_url)
                continue

            for candidate in self._extract_urls(response.text, base_url=section_url, domain=source_config.domain):
                if candidate in seen:
                    continue
                seen.add(candidate)
                urls.append(candidate)
                if limit is not None and len(urls) >= limit:
                    return urls

        return urls

    def _extract_urls(self, html: str, *, base_url: str, domain: str) -> list[str]:
        matches = self.ARTICLE_HREF_RE.findall(html)
        urls: list[str] = []
        for href in matches:
            absolute = urljoin(base_url, href)
            parsed = urlparse(absolute)
            if parsed.scheme in {"http", "https"} and domain in parsed.netloc:
                urls.append(absolute)
        return urls
