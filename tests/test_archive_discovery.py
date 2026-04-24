from __future__ import annotations

from datetime import date
import unittest

from app.ingestion.discovery import ArchiveDiscoveryService
from app.ingestion.fetcher import FetchResult
from app.ingestion.sources import SourceConfig


class MockFetcher:
    def __init__(self, payloads: dict[str, str]) -> None:
        self.payloads = payloads

    def fetch(self, url: str) -> FetchResult:
        if url not in self.payloads:
            raise RuntimeError(f"Unexpected URL: {url}")
        return FetchResult(url=url, status_code=200, text=self.payloads[url], content_type="text/html")


class ArchiveDiscoveryTests(unittest.TestCase):
    def test_lenta_archive_discovery_collects_article_links(self) -> None:
        archive_url = "https://lenta.ru/2026/04/23/"
        html = """
        <html>
          <body>
            <a href="/news/2026/04/23/iran/">Iran</a>
            <a href="/articles/2026/04/23/market/">Market</a>
            <a href="/rubrics/world/">Rubric</a>
          </body>
        </html>
        """
        service = ArchiveDiscoveryService(MockFetcher({archive_url: html}))
        source_config = SourceConfig(
            name="Lenta.ru",
            domain="lenta.ru",
            base_url="https://lenta.ru",
            parser_type="lenta",
            article_url_patterns=(
                r"^https?://lenta\.ru/news/\d{4}/\d{2}/\d{2}/[^/]+/?$",
                r"^https?://lenta\.ru/articles/\d{4}/\d{2}/\d{2}/[^/]+/?$",
            ),
            archive_url_template="https://lenta.ru/{year}/{month}/{day}/",
        )

        urls = service.discover_for_date(
            source_config,
            target_date=date(2026, 4, 23),
        )

        self.assertEqual(
            urls,
            [
                "https://lenta.ru/news/2026/04/23/iran/",
                "https://lenta.ru/articles/2026/04/23/market/",
            ],
        )

    def test_ria_archive_discovery_renders_ymd_url(self) -> None:
        archive_url = "https://ria.ru/20260423/"
        html = """
        <html>
          <body>
            <a href="https://ria.ru/20260423/iran-2088648192.html">Iran</a>
            <a href="https://ria.ru/world/">World</a>
          </body>
        </html>
        """
        service = ArchiveDiscoveryService(MockFetcher({archive_url: html}))
        source_config = SourceConfig(
            name="RIA",
            domain="ria.ru",
            base_url="https://ria.ru",
            parser_type="ria",
            article_url_patterns=(r"^https?://ria\.ru/\d{8}/[^/]+\.html$",),
            archive_url_template="https://ria.ru/{ymd}/",
        )

        discovered = service.discover_for_date_range(
            source_config,
            date_from=date(2026, 4, 23),
            date_to=date(2026, 4, 23),
            per_day_limit=10,
        )

        self.assertEqual(
            discovered[date(2026, 4, 23)],
            ["https://ria.ru/20260423/iran-2088648192.html"],
        )


if __name__ == "__main__":
    unittest.main()
