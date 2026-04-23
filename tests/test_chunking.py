from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from app.db.connection import create_connection
from app.db.schema import create_schema
from app.models import ArticleCreate, SourceCreate
from app.repositories import ArticleChunkRepository, ArticleRepository, SourceRepository
from app.services import ChunkingService


class ChunkingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path("tests") / ".tmp" / f"chunking-{uuid.uuid4().hex}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.temp_dir / "chunking.db"
        self.connection = create_connection(self.database_path)
        create_schema(self.connection)
        self.source_repo = SourceRepository(self.connection)
        self.article_repo = ArticleRepository(self.connection)
        self.chunk_repo = ArticleChunkRepository(self.connection)
        self.chunking_service = ChunkingService()

    def tearDown(self) -> None:
        self.connection.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_article(self):
        source = self.source_repo.create(
            SourceCreate(
                name="Test Source",
                domain=f"example-{uuid.uuid4().hex[:8]}.ru",
                base_url="https://example.ru",
                source_type="news_site",
            )
        )
        body_text = (
            "Первый абзац новости содержит достаточно информации для формирования чанка и описывает событие подробно.\n\n"
            "Второй абзац продолжает тему и добавляет контекст, который нужен для поиска по документу и retrieval.\n\n"
            "Третий абзац завершает материал и тоже должен попасть в один из чанков."
        )
        return self.article_repo.create_article(
            ArticleCreate(
                source_id=source.id,
                url=f"https://example.ru/articles/{uuid.uuid4().hex}",
                title="Тестовая статья",
                subtitle=None,
                body_text=body_text,
                published_at="2026-04-23T12:00:00",
                content_hash=uuid.uuid4().hex,
                word_count=35,
                is_canonical=True,
            )
        )

    def test_chunking_service_splits_article(self) -> None:
        article = self._create_article()

        chunks = self.chunking_service.chunk_article(article)

        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0].article_id, article.id)
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertTrue(chunks[0].chunk_text)

    def test_chunk_repository_saves_chunks(self) -> None:
        article = self._create_article()
        chunks = self.chunking_service.chunk_article(article)

        created = self.chunk_repo.create_many(chunks)
        loaded = self.chunk_repo.list_by_article_id(article.id)

        self.assertEqual(len(created), len(chunks))
        self.assertEqual(len(loaded), len(chunks))
        self.assertEqual(loaded[0].chunk_index, 0)
        self.assertTrue(loaded[0].chunk_text.startswith("Первый абзац"))


if __name__ == "__main__":
    unittest.main()
