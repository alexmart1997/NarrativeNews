from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from app.db.connection import create_connection
from app.db.schema import create_schema
from app.models import ArticleCreate, ChunkSearchResult, SourceCreate
from app.repositories import ArticleChunkRepository, ArticleRepository, SourceRepository
from app.services import BaseLLMClient, ChunkingService, RAGService


class MockLLMClient(BaseLLMClient):
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[ChunkSearchResult]]] = []

    def generate_answer(self, query: str, chunks: list[ChunkSearchResult]) -> str:
        self.calls.append((query, chunks))
        return f"Сводка по запросу: {query}. Использовано чанков: {len(chunks)}."


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
        self.mock_llm = MockLLMClient()
        self.rag_service = RAGService(self.chunk_repo, self.article_repo, llm_client=self.mock_llm)

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

    def test_rag_answer_returns_summary_and_source_articles(self) -> None:
        article_one = self._create_article(
            title="Статья 1",
            body_text="инфляция ускорилась в апреле. Экономисты обсуждают рост цен и реакцию рынка.",
            published_at="2026-04-20T10:00:00",
        )
        article_two = self._create_article(
            title="Статья 2",
            body_text="Банк России сообщил, что инфляция остается ключевым фактором денежно-кредитной политики.",
            published_at="2026-04-21T10:00:00",
        )

        result = self.rag_service.answer(
            query="инфляция",
            date_from="2026-04-01T00:00:00",
            date_to="2026-04-30T23:59:59",
            limit=5,
            include_debug_chunks=True,
        )

        self.assertIn("Сводка по запросу: инфляция", result.summary_text)
        self.assertGreaterEqual(len(result.source_articles), 2)
        self.assertEqual({article.id for article in result.source_articles[:2]}, {article_one.id, article_two.id})
        self.assertIsNotNone(result.top_chunks)

    def test_rag_answer_passes_retrieval_results_to_llm(self) -> None:
        self._create_article(
            title="Статья 1",
            body_text="Инфляция ускорилась в апреле и остается важной темой для аналитиков.",
            published_at="2026-04-20T10:00:00",
        )

        self.rag_service.answer(
            query="Инфляция",
            date_from="2026-04-01T00:00:00",
            date_to="2026-04-30T23:59:59",
            limit=3,
        )

        self.assertEqual(len(self.mock_llm.calls), 1)
        query, chunks = self.mock_llm.calls[0]
        self.assertEqual(query, "Инфляция")
        self.assertGreaterEqual(len(chunks), 1)
        self.assertTrue(all(isinstance(chunk, ChunkSearchResult) for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
