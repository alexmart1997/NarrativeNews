from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path
import shutil
import uuid

from app.db.connection import create_connection
from app.db.schema import create_schema
from app.models import ArticleCreate, SourceCreate
from app.repositories import ArticleRepository, NarrativeAnalysisRepository, SourceRepository
from app.utils.text import estimate_word_count


class RepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path("tests") / ".tmp" / f"repository-{uuid.uuid4().hex}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.temp_dir / "repository.db"
        self.connection = create_connection(self.database_path)
        create_schema(self.connection)
        self.source_repo = SourceRepository(self.connection)
        self.article_repo = ArticleRepository(self.connection)
        self.narrative_repo = NarrativeAnalysisRepository(self.connection)

    def tearDown(self) -> None:
        self.connection.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_source(self) -> int:
        source = self.source_repo.create(
            SourceCreate(
                name="ТАСС",
                domain="tass.ru",
                base_url="https://tass.ru",
                source_type="news_agency",
            )
        )
        return source.id

    def create_article(
        self,
        source_id: int,
        *,
        url: str,
        published_at: str,
        canonical: bool = True,
    ):
        return self.article_repo.create_article(
            ArticleCreate(
                source_id=source_id,
                url=url,
                title="Тестовая статья",
                subtitle=None,
                body_text="Россия обсуждает новый пример тестовой новости.",
                published_at=published_at,
                author="Автор",
                category="society",
                content_hash=f"hash-{url.rsplit('/', 1)[-1]}",
                word_count=estimate_word_count("Россия обсуждает новый пример тестовой новости."),
                is_canonical=canonical,
            )
        )

    def test_create_and_read_source(self) -> None:
        created = self.source_repo.create(
            SourceCreate(
                name="Интерфакс",
                domain="interfax.ru",
                base_url="https://www.interfax.ru",
                source_type="news_agency",
            )
        )

        loaded = self.source_repo.get_by_id(created.id)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.domain, "interfax.ru")
        self.assertTrue(loaded.is_active)

    def test_create_and_read_article(self) -> None:
        source_id = self.create_source()

        created = self.article_repo.create_article(
            ArticleCreate(
                source_id=source_id,
                url="https://tass.ru/obschestvo/1",
                title="Заголовок",
                subtitle="Подзаголовок",
                body_text="Тестовый текст новости для проверки репозитория статей.",
                published_at="2026-04-20T10:00:00",
                author="Редакция",
                category="society",
                content_hash="hash-001",
                word_count=estimate_word_count("Тестовый текст новости для проверки репозитория статей."),
            )
        )

        loaded = self.article_repo.get_article_by_url("https://tass.ru/obschestvo/1")

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.id, created.id)
        self.assertEqual(loaded.title, "Заголовок")
        self.assertTrue(loaded.is_canonical)

    def test_foreign_key_integrity_for_articles(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            self.article_repo.create_article(
                ArticleCreate(
                    source_id=9999,
                    url="https://invalid.test/article",
                    title="Broken",
                    subtitle=None,
                    body_text="No source should fail.",
                    published_at="2026-04-20T10:00:00",
                    content_hash="invalid-hash",
                )
            )

    def test_list_articles_by_date_range(self) -> None:
        source_id = self.create_source()

        self.create_article(
            source_id,
            url="https://tass.ru/a1",
            published_at="2026-04-18T08:00:00",
            canonical=True,
        )
        self.create_article(
            source_id,
            url="https://tass.ru/a2",
            published_at="2026-04-19T08:00:00",
            canonical=False,
        )
        self.create_article(
            source_id,
            url="https://tass.ru/a3",
            published_at="2026-04-21T08:00:00",
            canonical=True,
        )

        loaded = self.article_repo.list_articles_by_date_range(
            "2026-04-18T00:00:00",
            "2026-04-20T23:59:59",
        )
        canonical = self.article_repo.list_canonical_articles_by_date_range(
            "2026-04-18T00:00:00",
            "2026-04-22T00:00:00",
        )

        self.assertEqual(
            [article.url for article in loaded],
            ["https://tass.ru/a1", "https://tass.ru/a2"],
        )
        self.assertEqual(
            [article.url for article in canonical],
            ["https://tass.ru/a1", "https://tass.ru/a3"],
        )

    def test_save_and_load_narrative_snapshot_run(self) -> None:
        saved = self.narrative_repo.save_run(
            source_domains_key="ria.ru",
            date_from="20260101T0000",
            date_to="20260430T2359",
            payload_json='{"topics": [], "clusters": []}',
            status="completed",
            documents_count=10,
            topics_count=2,
            frames_count=5,
            clusters_count=1,
            labels_count=1,
            assignments_count=4,
            dynamics_count=1,
        )

        loaded = self.narrative_repo.get_latest_run(
            source_domains_key="ria.ru",
            date_from="2026-01-01T00:00:00",
            date_to="2026-04-30T23:59:59",
        )
        payload = self.narrative_repo.get_latest_payload(
            source_domains_key="ria.ru",
            date_from="2026-01-01T00:00:00",
            date_to="2026-04-30T23:59:59",
        )

        self.assertIsNotNone(loaded)
        self.assertEqual(saved.id, loaded.id)
        self.assertEqual(loaded.documents_count, 10)
        self.assertEqual(payload, {"topics": [], "clusters": []})


if __name__ == "__main__":
    unittest.main()
