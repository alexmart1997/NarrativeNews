from __future__ import annotations

import hashlib
import shutil
import unittest
import uuid
from pathlib import Path

from app.db.connection import create_connection
from app.db.schema import create_schema
from app.ingestion.fetcher import FetchResult
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.sources import SourceConfig
from app.models import ArticleCreate, SourceCreate
from app.repositories import ArticleChunkRepository, ArticleRepository, SourceRepository


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
        self.chunk_repo = ArticleChunkRepository(self.connection)

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
        fetcher = MockFetcher({rss_url: rss_payload, article_url: self._read_sample("lenta_article.html")})
        pipeline = IngestionPipeline(
            fetcher=fetcher,
            source_repository=self.source_repo,
            article_repository=self.article_repo,
            article_chunk_repository=self.chunk_repo,
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
        saved_article = self.article_repo.get_article_by_url("https://lenta.ru/news/2026/04/23/example")
        saved_chunks = self.chunk_repo.list_by_article_id(saved_article.id if saved_article else -1)

        self.assertEqual(result.discovered_urls, 1)
        self.assertEqual(result.saved_articles, 1)
        self.assertEqual(result.skipped_duplicates, 0)
        self.assertIsNotNone(saved_article)
        self.assertEqual(saved_article.title, "Лента сообщила о новой инициативе")
        self.assertTrue(saved_article.is_canonical)
        self.assertIsNone(saved_article.duplicate_group_id)
        self.assertGreaterEqual(len(saved_chunks), 1)

    def test_pipeline_skips_duplicate_by_url(self) -> None:
        source = self.source_repo.create(
            SourceCreate(
                name="Lenta.ru",
                domain="lenta.ru",
                base_url="https://lenta.ru",
                source_type="news_site",
            )
        )
        self.article_repo.create_article(
            ArticleCreate(
                source_id=source.id,
                url="https://lenta.ru/news/2026/04/23/example",
                title="Старая статья",
                subtitle=None,
                body_text="Первый абзац новости с достаточной длиной для тестирования парсинга и сохранения текста статьи в системе.",
                published_at="2026-04-23T10:15:00+03:00",
                content_hash="existing-hash-url",
                word_count=14,
                is_canonical=True,
            )
        )
        rss_url = "https://lenta.ru/rss"
        article_url = "https://lenta.ru/news/2026/04/23/example/"
        rss_payload = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item><link>{article_url}</link></item>
  </channel>
</rss>
"""
        fetcher = MockFetcher({rss_url: rss_payload, article_url: self._read_sample("lenta_article.html")})
        pipeline = IngestionPipeline(
            fetcher=fetcher,
            source_repository=self.source_repo,
            article_repository=self.article_repo,
            article_chunk_repository=self.chunk_repo,
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
        duplicate_rows = self.connection.execute("SELECT * FROM article_duplicates").fetchall()

        self.assertEqual(result.saved_articles, 0)
        self.assertEqual(result.skipped_existing, 1)
        self.assertEqual(result.skipped_duplicates, 1)
        self.assertEqual(len(duplicate_rows), 1)
        self.assertEqual(duplicate_rows[0]["duplicate_type"], "url")

    def test_pipeline_skips_duplicate_by_hash(self) -> None:
        source = self.source_repo.create(
            SourceCreate(
                name="РИА Новости",
                domain="ria.ru",
                base_url="https://ria.ru",
                source_type="news_site",
            )
        )
        canonical_body = (
            "Первый абзац статьи РИА с нормальной длиной и содержанием, которое должно быть сохранено в базу данных.\n\n"
            "Второй абзац развивает тему и тоже должен оказаться в основном тексте без служебных вставок."
        )
        self.article_repo.create_article(
            ArticleCreate(
                source_id=source.id,
                url="https://ria.ru/20260422/original.html",
                title="РИА: оригинальная статья",
                subtitle=None,
                body_text=canonical_body,
                published_at="2026-04-22T08:30:00+03:00",
                content_hash=hashlib.sha256(canonical_body.encode("utf-8")).hexdigest(),
                word_count=27,
                is_canonical=True,
            )
        )
        rss_url = "https://ria.ru/export/rss2/index.xml"
        duplicate_url = "https://ria.ru/20260422/duplicate.html"
        rss_payload = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item><link>{duplicate_url}</link></item>
  </channel>
</rss>
"""
        fetcher = MockFetcher({rss_url: rss_payload, duplicate_url: self._read_sample("ria_article.html")})
        pipeline = IngestionPipeline(
            fetcher=fetcher,
            source_repository=self.source_repo,
            article_repository=self.article_repo,
            article_chunk_repository=self.chunk_repo,
            min_body_length=80,
        )
        source_config = SourceConfig(
            name="РИА Новости",
            domain="ria.ru",
            base_url="https://ria.ru",
            rss_urls=(rss_url,),
            section_urls=(),
            parser_type="ria",
        )

        result = pipeline.run_once(source_config, limit=5)
        duplicate_rows = self.connection.execute("SELECT * FROM article_duplicates").fetchall()

        self.assertEqual(result.saved_articles, 0)
        self.assertEqual(result.skipped_existing, 0)
        self.assertEqual(result.skipped_duplicates, 1)
        self.assertEqual(len(duplicate_rows), 1)
        self.assertEqual(duplicate_rows[0]["duplicate_type"], "content_hash")


if __name__ == "__main__":
    unittest.main()
