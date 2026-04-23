from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from app.db.connection import create_connection
from app.db.schema import create_schema
from app.models import ArticleCreate, SourceCreate
from app.repositories import ArticleChunkRepository, ArticleRepository, SourceRepository
from app.services import ChunkingService, RAGService


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path("tests") / ".tmp" / f"retrieval-{uuid.uuid4().hex}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.temp_dir / "retrieval.db"
        self.connection = create_connection(self.database_path)
        create_schema(self.connection)
        self.source_repo = SourceRepository(self.connection)
        self.article_repo = ArticleRepository(self.connection)
        self.chunk_repo = ArticleChunkRepository(self.connection)
        self.chunking_service = ChunkingService()
        self.rag_service = RAGService(self.chunk_repo, self.article_repo)

    def tearDown(self) -> None:
        self.connection.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_article(self, *, title: str, body_text: str, published_at: str, is_canonical: bool = True):
        source = self.source_repo.get_by_domain("example.ru")
        if source is None:
            source = self.source_repo.create(
                SourceCreate(
                    name="Example",
                    domain="example.ru",
                    base_url="https://example.ru",
                    source_type="news_site",
                )
            )
        article = self.article_repo.create_article(
            ArticleCreate(
                source_id=source.id,
                url=f"https://example.ru/articles/{uuid.uuid4().hex}",
                title=title,
                subtitle=None,
                body_text=body_text,
                published_at=published_at,
                content_hash=uuid.uuid4().hex,
                word_count=50,
                is_canonical=is_canonical,
            )
        )
        self.chunk_repo.create_many(self.chunking_service.chunk_article(article))
        return article

    def test_retrieval_returns_matching_chunks_for_canonical_articles(self) -> None:
        matching_article = self._create_article(
            title="Экономика",
            body_text=(
                "Центробанк сообщил, что инфляция ускорилась в апреле и описал реакцию рынка.\n\n"
                "Аналитики обсуждают инфляцию и денежно-кредитную политику."
            ),
            published_at="2026-04-20T10:00:00",
            is_canonical=True,
        )
        self._create_article(
            title="Неканоническая статья",
            body_text="Инфляция упоминается и здесь, но статья неканоническая.",
            published_at="2026-04-20T12:00:00",
            is_canonical=False,
        )
        self._create_article(
            title="Другая дата",
            body_text="Инфляция была в прошлом месяце, но дата статьи вне диапазона.",
            published_at="2026-03-01T12:00:00",
            is_canonical=True,
        )

        result = self.rag_service.search(
            query="инфляция",
            date_from="2026-04-01T00:00:00",
            date_to="2026-04-30T23:59:59",
            limit=5,
        )

        self.assertGreaterEqual(len(result.chunks), 1)
        self.assertEqual(result.articles[0].id, matching_article.id)
        self.assertIn("инфляция", result.chunks[0].chunk_text.lower())


if __name__ == "__main__":
    unittest.main()
