from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from app.db.connection import create_connection
from app.db.schema import create_schema
from app.models import ArticleCreate, SourceCreate
from app.repositories import ArticleChunkRepository, ArticleRepository, SourceRepository
from app.services import BaseEmbeddingClient, BaseLLMClient, ChunkingService, EmbeddingIndexService, RAGService


class MockLLMClient(BaseLLMClient):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.summary_text = "Краткая сводка по найденным материалам."
        self.reranked_ids: list[int] = []

    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if "ranked_chunk_ids" in prompt:
            return '{"ranked_chunk_ids": %s}' % self.reranked_ids
        return self.summary_text


class MockEmbeddingClient(BaseEmbeddingClient):
    def __init__(self) -> None:
        self._model_name = "mock-embed"

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_text(self, text: str) -> list[float]:
        lowered = text.lower()
        if "иран" in lowered or "тегеран" in lowered:
            return [1.0, 0.0]
        if "украин" in lowered or "киев" in lowered:
            return [0.0, 1.0]
        return [0.2, 0.2]


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
        self.mock_embedding = MockEmbeddingClient()
        self.embedding_index_service = EmbeddingIndexService(self.chunk_repo, self.mock_embedding)
        self.rag_service = RAGService(
            self.chunk_repo,
            self.article_repo,
            llm_client=self.mock_llm,
            embedding_client=self.mock_embedding,
            hybrid_limit=12,
            rerank_limit=5,
        )

    def tearDown(self) -> None:
        self.connection.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_article(
        self,
        *,
        title: str,
        body_text: str,
        published_at: str,
        is_canonical: bool = True,
        source_domain: str = "example.ru",
        source_name: str = "Example",
    ):
        source = self.source_repo.get_by_domain(source_domain)
        if source is None:
            source = self.source_repo.create(
                SourceCreate(
                    name=source_name,
                    domain=source_domain,
                    base_url=f"https://{source_domain}",
                    source_type="news_site",
                )
            )
        article = self.article_repo.create_article(
            ArticleCreate(
                source_id=source.id,
                url=f"https://{source_domain}/articles/{uuid.uuid4().hex}",
                title=title,
                subtitle=None,
                body_text=body_text,
                published_at=published_at,
                content_hash=uuid.uuid4().hex,
                word_count=50,
                is_canonical=is_canonical,
            )
        )
        chunks = self.chunk_repo.create_many(self.chunking_service.chunk_article(article))
        self.embedding_index_service.index_chunks(chunks)
        return article

    def test_lexical_retrieval_returns_matching_chunks_for_canonical_articles(self) -> None:
        matching_article = self._create_article(
            title="Экономика",
            body_text="Инфляция ускорилась в апреле. Экономисты обсуждают спрос и реакцию рынка.",
            published_at="2026-04-20T10:00:00",
            is_canonical=True,
        )
        self._create_article(
            title="Неканоническая статья",
            body_text="Инфляция упоминается и здесь, но статья неканоническая.",
            published_at="2026-04-20T12:00:00",
            is_canonical=False,
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

    def test_hybrid_retrieval_can_find_semantic_match_via_embeddings(self) -> None:
        target_article = self._create_article(
            title="Переговоры по сделке",
            body_text="Тегеран выступил с новым заявлением по ядерной сделке и региональной безопасности.",
            published_at="2026-04-21T10:00:00",
        )
        self._create_article(
            title="Новости Украины",
            body_text="Киев сообщил о новых переговорах и развитии ситуации на фронте.",
            published_at="2026-04-21T11:00:00",
        )

        result = self.rag_service.search(
            query="иран",
            date_from="2026-04-01T00:00:00",
            date_to="2026-04-30T23:59:59",
            limit=5,
        )

        self.assertGreaterEqual(len(result.chunks), 1)
        self.assertEqual(result.articles[0].id, target_article.id)
        self.assertGreater(result.chunks[0].vector_score, 0.7)

    def test_topic_filter_excludes_irrelevant_articles_when_direct_matches_exist(self) -> None:
        target_article = self._create_article(
            title="Иран сделал заявление",
            body_text="Иран сделал новое заявление по ядерной сделке и региональной безопасности.",
            published_at="2026-04-21T10:00:00",
        )
        self._create_article(
            title="ДТП в Москве",
            body_text="Водитель каршеринга сбил двух женщин в Москве во время побега от полиции.",
            published_at="2026-04-21T11:00:00",
        )

        result = self.rag_service.answer(
            query="иран",
            date_from="2026-04-01T00:00:00",
            date_to="2026-04-30T23:59:59",
            limit=5,
            include_debug_chunks=True,
        )

        self.assertEqual(len(result.source_articles), 1)
        self.assertEqual(result.source_articles[0].id, target_article.id)
        self.assertTrue(all("иран" in chunk.chunk_text.lower() or "иран" in chunk.article_title.lower() for chunk in (result.top_chunks or [])))

    def test_natural_language_query_is_normalized_to_topic_terms(self) -> None:
        target_article = self._create_article(
            title="Иран сделал заявление",
            body_text="Иран сделал новое заявление по ядерной сделке и региональной безопасности.",
            published_at="2026-04-21T10:00:00",
        )
        self._create_article(
            title="Новости Москвы",
            body_text="В Москве произошло ДТП с участием каршеринга и полицейской погони.",
            published_at="2026-04-21T11:00:00",
        )

        result = self.rag_service.answer(
            query="Что с Ираном",
            date_from="2026-04-01T00:00:00",
            date_to="2026-04-30T23:59:59",
            limit=5,
            include_debug_chunks=True,
        )

        self.assertEqual(len(result.source_articles), 1)
        self.assertEqual(result.source_articles[0].id, target_article.id)

    def test_rag_answer_uses_reranked_chunks_and_returns_articles(self) -> None:
        first_article = self._create_article(
            title="Статья 1",
            body_text="Иран сообщил о новых переговорах и уточнил позицию по сделке.",
            published_at="2026-04-20T10:00:00",
        )
        second_article = self._create_article(
            title="Статья 2",
            body_text="Тегеран также сделал отдельное заявление о региональной безопасности.",
            published_at="2026-04-21T10:00:00",
        )

        initial_chunks = self.rag_service.search_chunks(
            query="иран",
            date_from="2026-04-01T00:00:00",
            date_to="2026-04-30T23:59:59",
            limit=5,
        )
        self.mock_llm.reranked_ids = [initial_chunks[-1].chunk_id, initial_chunks[0].chunk_id]

        result = self.rag_service.answer(
            query="иран",
            date_from="2026-04-01T00:00:00",
            date_to="2026-04-30T23:59:59",
            limit=5,
            include_debug_chunks=True,
        )

        self.assertEqual(result.summary_text, "Краткая сводка по найденным материалам.")
        self.assertGreaterEqual(len(result.source_articles), 2)
        self.assertEqual({article.id for article in result.source_articles[:2]}, {first_article.id, second_article.id})
        self.assertIsNotNone(result.top_chunks)
        self.assertTrue(any("ranked_chunk_ids" in str(call["prompt"]) for call in self.mock_llm.calls))

    def test_rag_answer_falls_back_when_llm_returns_non_russian_text(self) -> None:
        self.mock_llm.summary_text = "您好！这里没有相关信息。"
        self._create_article(
            title="Статья 1",
            body_text="Иран сообщил о новых переговорах и уточнил позицию по сделке.",
            published_at="2026-04-20T10:00:00",
        )

        result = self.rag_service.answer(
            query="иран",
            date_from="2026-04-01T00:00:00",
            date_to="2026-04-30T23:59:59",
            limit=5,
        )

        self.assertIn("Иран", result.summary_text)

    def test_rag_can_filter_by_source_domain(self) -> None:
        ria_article = self._create_article(
            title="Инфляция в РИА",
            body_text="Инфляция ускорилась, РИА приводит подробности по экономике и ценам.",
            published_at="2026-04-22T10:00:00",
            source_domain="ria.ru",
            source_name="РИА Новости",
        )
        self._create_article(
            title="Инфляция в Ленте",
            body_text="Инфляция ускорилась, Лента публикует похожий обзор и комментарии.",
            published_at="2026-04-22T11:00:00",
            source_domain="lenta.ru",
            source_name="Лента.ру",
        )

        result = self.rag_service.answer(
            query="инфляция",
            date_from="2026-04-01T00:00:00",
            date_to="2026-04-30T23:59:59",
            limit=5,
            source_domains=["ria.ru"],
        )

        self.assertEqual(len(result.source_articles), 1)
        self.assertEqual(result.source_articles[0].id, ria_article.id)


if __name__ == "__main__":
    unittest.main()
