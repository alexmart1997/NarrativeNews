from __future__ import annotations

import logging
import re
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

            for item_url in self._extract_urls(response.text, source_config):
                if item_url in seen:
                    continue
                seen.add(item_url)
                urls.append(item_url)
                if limit is not None and len(urls) >= limit:
                    return urls

        return urls

    def _extract_urls(self, rss_text: str, source_config: SourceConfig) -> list[str]:
        root = ET.fromstring(rss_text)
        candidates: list[str] = []
        for entry in root.iter():
            tag_name = self._local_name(entry.tag)
            if tag_name not in {"item", "entry"}:
                continue

            url = self._extract_entry_url(entry)
            if url and self._is_article_url(url, source_config):
                candidates.append(url)
        return candidates

    @classmethod
    def _extract_entry_url(cls, entry: ET.Element) -> str | None:
        for child in entry:
            tag_name = cls._local_name(child.tag)
            if tag_name != "link":
                continue

            href = child.attrib.get("href")
            if href:
                return href.strip()

            if child.text and child.text.strip():
                return child.text.strip()

        return None

    @staticmethod
    def _local_name(tag: str) -> str:
        if "}" in tag:
            return tag.rsplit("}", 1)[-1].lower()
        return tag.lower()

    @staticmethod
    def _is_article_url(url: str, source_config: SourceConfig) -> bool:
        parsed = urlparse(url)
        if not (parsed.scheme and parsed.netloc and source_config.domain in parsed.netloc):
            return False
        if parsed.path in {"", "/"}:
            return False
        normalized_base = source_config.base_url.rstrip("/")
        normalized_url = url.rstrip("/")
        if normalized_url == normalized_base:
            return False
        if source_config.article_url_patterns:
            return any(re.match(pattern, url) for pattern in source_config.article_url_patterns)
        return True
