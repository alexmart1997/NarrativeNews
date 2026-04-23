from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from app.db.connection import create_connection
from app.db.schema import create_schema
from app.ingestion.fetcher import FetchResult
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.sources import SourceConfig
from app.repositories import ArticleRepository, SourceRepository


class MockFetcher:
    def __init__(self, payloads: dict[str, str]) -> None:
        self.payloads = payloads

    def fetch(self, url: str) -> FetchResult:
        if url not in self.payloads:
            raise RuntimeError(f"Unexpected URL: {url}")
        payload = self.payloads[url]
        content_type = "application/rss+xml" if url.endswith(".xml") else "text/html"
        return FetchResult(url=url, status_code=200, text=payload, content_type=content_type)


class IngestionPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path("tests") / ".tmp" / f"pipeline-{uuid.uuid4().hex}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.temp_dir / "pipeline.db"
        self.connection = create_connection(self.database_path)
        create_schema(self.connection)
        self.source_repo = SourceRepository(self.connection)
        self.article_repo = ArticleRepository(self.connection)

    def tearDown(self) -> None:
        self.connection.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _read_sample(self, filename: str) -> str:
        return (Path("tests") / "samples" / filename).read_text(encoding="utf-8")

    def test_pipeline_ingests_one_article_from_rss(self) -> None:
        rss_url = "https://lenta.ru/rss"
        article_url = "https://lenta.ru/news/2026/04/23/example/"
        rss_payload = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Lenta</title>
    <item>
      <link>{article_url}</link>
    </item>
  </channel>
</rss>
"""
        fetcher = MockFetcher(
            {
                rss_url: rss_payload,
                article_url: self._read_sample("lenta_article.html"),
            }
        )
        pipeline = IngestionPipeline(
            fetcher=fetcher,
            source_repository=self.source_repo,
            article_repository=self.article_repo,
            min_body_length=80,
        )
        source_config = SourceConfig(
            name="Lenta.ru",
            domain="lenta.ru",
            base_url="https://lenta.ru",
            rss_urls=(rss_url,),
            section_urls=(),
            parser_type="lenta",
        )

        result = pipeline.run_once(source_config, limit=5)
        saved_article = self.article_repo.get_article_by_url(article_url)

        self.assertEqual(result.discovered_urls, 1)
        self.assertEqual(result.saved_articles, 1)
        self.assertIsNotNone(saved_article)
        self.assertEqual(saved_article.title, "Лента сообщила о новой инициативе")
        self.assertTrue(saved_article.is_canonical)
        self.assertIsNone(saved_article.duplicate_group_id)


if __name__ == "__main__":
    unittest.main()
