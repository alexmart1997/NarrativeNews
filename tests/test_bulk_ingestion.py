from __future__ import annotations

import shutil
import unittest
import uuid
from datetime import date
from pathlib import Path

from app.db.connection import create_connection
from app.db.schema import create_schema
from app.ingestion.discovery import ArchiveDiscoveryService
from app.ingestion.fetcher import FetchResult
from app.ingestion.pipeline import BulkIngestionService, IngestionPipeline
from app.ingestion.sources import SourceConfig
from app.repositories import ArticleChunkRepository, ArticleRepository, SourceRepository


class MockFetcher:
    def __init__(self, payloads: dict[str, str]) -> None:
        self.payloads = payloads

    def fetch(self, url: str) -> FetchResult:
        if url not in self.payloads:
            raise RuntimeError(f"Unexpected URL: {url}")
        payload = self.payloads[url]
        content_type = "text/html"
        return FetchResult(url=url, status_code=200, text=payload, content_type=content_type)


class BulkIngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path("tests") / ".tmp" / f"bulk-{uuid.uuid4().hex}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.temp_dir / "bulk.db"
        self.connection = create_connection(self.database_path)
        create_schema(self.connection)
        self.source_repo = SourceRepository(self.connection)
        self.article_repo = ArticleRepository(self.connection)
        self.chunk_repo = ArticleChunkRepository(self.connection)

    def tearDown(self) -> None:
        self.connection.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _read_sample(self, filename: str) -> str:
        return (Path("tests") / "samples" / filename).read_text(encoding="utf-8")

    def test_bulk_ingestion_runs_over_archive_range(self) -> None:
        day_one_archive = "https://lenta.ru/2026/04/22/"
        day_two_archive = "https://lenta.ru/2026/04/23/"
        article_one = "https://lenta.ru/news/2026/04/22/one/"
        article_two = "https://lenta.ru/news/2026/04/23/two/"
        article_duplicate = "https://lenta.ru/news/2026/04/22/one/"
        article_two_html = """
        <html>
          <head>
            <meta property="article:published_time" content="2026-04-23T12:30:00+03:00" />
          </head>
          <body>
            <a class="topic-header__rubric">Экономика</a>
            <h1 class="topic-body__title">Новый сценарий переговоров</h1>
            <div class="topic-body__title-topic">Второй материал для bulk ingestion</div>
            <span class="topic-authors__name">Редакция</span>
            <section class="topic-page__content">
              <p>Переговоры продолжились в новом формате, и участники представили обновленный план действий на ближайшие недели.</p>
              <p>Отдельно обсуждались экономические последствия и влияние новых договоренностей на региональную торговлю.</p>
            </section>
          </body>
        </html>
        """
        fetcher = MockFetcher(
            {
                day_one_archive: f'<a href="{article_one}">One</a>',
                day_two_archive: f'<a href="{article_two}">Two</a><a href="{article_duplicate}">Dup</a>',
                article_one: self._read_sample("lenta_article.html"),
                article_two: article_two_html,
            }
        )
        pipeline = IngestionPipeline(
            fetcher=fetcher,
            source_repository=self.source_repo,
            article_repository=self.article_repo,
            article_chunk_repository=self.chunk_repo,
            min_body_length=80,
        )
        service = BulkIngestionService(
            pipeline=pipeline,
            archive_discovery_service=ArchiveDiscoveryService(fetcher),
        )
        source_config = SourceConfig(
            name="Lenta.ru",
            domain="lenta.ru",
            base_url="https://lenta.ru",
            parser_type="lenta",
            article_url_patterns=(r"^https?://lenta\.ru/news/\d{4}/\d{2}/\d{2}/[^/]+/?$",),
            archive_url_template="https://lenta.ru/{year}/{month}/{day}/",
        )

        result = service.run_for_date_range(
            source_config,
            date_from=date(2026, 4, 22),
            date_to=date(2026, 4, 23),
            per_day_limit=20,
        )

        self.assertEqual(result.days_processed, 2)
        self.assertEqual(result.discovered_urls, 3)
        self.assertEqual(result.saved_articles, 2)
        self.assertEqual(result.skipped_existing, 1)
        self.assertEqual(result.skipped_duplicates, 1)
        self.assertIsNotNone(self.article_repo.get_article_by_url("https://lenta.ru/news/2026/04/22/one"))
        self.assertIsNotNone(self.article_repo.get_article_by_url("https://lenta.ru/news/2026/04/23/two"))


if __name__ == "__main__":
    unittest.main()
