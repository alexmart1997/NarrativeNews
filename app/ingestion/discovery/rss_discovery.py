from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

from app.ingestion.fetcher import FetchError, HttpFetcher
from app.ingestion.sources import SourceConfig

logger = logging.getLogger(__name__)


class RSSDiscoveryService:
    def __init__(self, fetcher: HttpFetcher) -> None:
        self.fetcher = fetcher

    def discover(self, source_config: SourceConfig, limit: int | None = None) -> list[str]:
        urls: list[str] = []
        seen: set[str] = set()

        for rss_url in source_config.rss_urls:
            try:
                response = self.fetcher.fetch(rss_url)
            except FetchError:
                logger.exception("Failed to fetch RSS feed %s", rss_url)
                continue

            for item_url in self._extract_urls(response.text, source_config.domain):
                if item_url in seen:
                    continue
                seen.add(item_url)
                urls.append(item_url)
                if limit is not None and len(urls) >= limit:
                    return urls

        return urls

    def _extract_urls(self, rss_text: str, domain: str) -> list[str]:
        root = ET.fromstring(rss_text)
        candidates: list[str] = []
        for element in root.iter():
            if element.tag.lower().endswith("link") and element.text:
                url = element.text.strip()
                if url and self._is_article_url(url, domain):
                    candidates.append(url)
        return candidates

    @staticmethod
    def _is_article_url(url: str, domain: str) -> bool:
        parsed = urlparse(url)
        return bool(parsed.scheme and parsed.netloc and domain in parsed.netloc)
