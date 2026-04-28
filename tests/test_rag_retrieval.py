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
from app.services.reranker import BaseChunkReranker


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
        if "инфляц" in lowered or "цен" in lowered or "подорож" in lowered:
            return [0.8, 0.2]
        if "рубл" in lowered or "курс" in lowered or "экономик" in lowered:
            return [0.3, 0.7]
        return [0.2, 0.2]


class MockReranker(BaseChunkReranker):
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def score(self, query: str, candidates: list) -> list[float]:
        self.calls.append((query, len(candidates)))
        scores: list[float] = []
        for candidate in candidates:
            text = f"{candidate.article_title} {candidate.chunk_text}".lower()
            if "тегеран" in text or "иран" in text:
                scores.append(0.95)
            else:
                scores.append(0.10)
        return scores


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
        self.mock_reranker = MockReranker()
        self.embedding_index_service = EmbeddingIndexService(self.chunk_repo, self.mock_embedding)
        self.rag_service = RAGService(
            self.chunk_repo,
            self.article_repo,
            llm_client=self.mock_llm,
            embedding_client=self.mock_embedding,
            reranker=self.mock_reranker,
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
        self.assertIn("инфляц", result.chunks[0].chunk_text.lower())

    def test_hybrid_retrieval_can_find_semantic_match_via_embeddings(self) -> None:
        target_article = self._create_article(
            title="Переговоры по сделке",
            body_text="Тегеран выступил с новым заявлением по ядерной сделке и региональной безопасности.",
            published_at="2026-04-21T10:00:00",
        )
        self._create_article(
            title="Новости Украины",
            body_text="Киев обсуждает новый пакет помощи и ситуацию на фронте.",
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
            title="Доп в Москве",
            body_text="Водитель каршеринга был задержан в Москве после побега от полиции.",
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
        self.assertTrue(
            all(
                "иран" in chunk.chunk_text.lower() or "иран" in chunk.article_title.lower()
                for chunk in (result.top_chunks or [])
            )
        )

    def test_natural_language_query_is_normalized_to_topic_terms(self) -> None:
        target_article = self._create_article(
            title="Иран сделал заявление",
            body_text="Иран сделал новое заявление по ядерной сделке и региональной безопасности.",
            published_at="2026-04-21T10:00:00",
        )
        self._create_article(
            title="Новости Москвы",
            body_text="В Москве произошло ДТП с участием каршеринга и полиции.",
            published_at="2026-04-21T11:00:00",
        )

        result = self.rag_service.answer(
            query="что с Ираном",
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
            body_text="Иран сообщил о новых переговорах по сделке и обозначил позицию по безопасности.",
            published_at="2026-04-20T10:00:00",
        )
        second_article = self._create_article(
            title="Статья 2",
            body_text="Иран также сделал отдельное заявление о региональной безопасности и дипломатии.",
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
        self.mock_llm.summary_text = "plain english output only"
        self._create_article(
            title="Статья 1",
            body_text="Иран сообщил о новых переговорах по сделке и обозначил позицию по безопасности.",
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
            body_text="Инфляция ускорилась, Лента публикует похожий обзор по экономике.",
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

    def test_rag_supports_iso_date_filters_for_compact_published_at(self) -> None:
        article = self._create_article(
            title="Иран и переговоры",
            body_text="Иран сделал новое заявление по переговорам и региональной безопасности.",
            published_at="20260421T1015",
            source_domain="ria.ru",
            source_name="РИА Новости",
        )

        result = self.rag_service.answer(
            query="иран",
            date_from="2026-04-01T00:00:00",
            date_to="2026-04-30T23:59:59",
            limit=5,
            source_domains=["ria.ru"],
        )

        self.assertEqual(len(result.source_articles), 1)
        self.assertEqual(result.source_articles[0].id, article.id)

    def test_rag_prefers_anchor_topic_for_broad_economic_query(self) -> None:
        inflation_article = self._create_article(
            title="Инфляция в России ускорилась в марте",
            body_text=(
                "Инфляция в России ускорилась в марте, а рост потребительских цен затронул продукты, "
                "услуги и тарифы. Аналитики связывают подорожание товаров с ослаблением рубля и "
                "издержками производителей."
            ),
            published_at="2026-04-22T10:00:00",
            source_domain="ria.ru",
            source_name="РИА Новости",
        )
        self._create_article(
            title="Для российской экономики назвали выгодный курс рубля",
            body_text=(
                "Эксперт заявил, что курс рубля влияет на доходы бюджета, устойчивость экспортеров "
                "и состояние российской экономики. Материал посвящен курсу валют и финансовым рынкам."
            ),
            published_at="2026-04-22T11:00:00",
            source_domain="ria.ru",
            source_name="РИА Новости",
        )

        result = self.rag_service.answer(
            query="инфляция в россии",
            date_from="2026-04-01T00:00:00",
            date_to="2026-04-30T23:59:59",
            limit=5,
            source_domains=["ria.ru"],
        )

        self.assertGreaterEqual(len(result.source_articles), 1)
        self.assertEqual(result.source_articles[0].id, inflation_article.id)

    def test_rag_penalizes_boilerplate_chunks_even_when_title_matches(self) -> None:
        useful_article = self._create_article(
            title="Иран сделал заявление по ядерной сделке",
            body_text=(
                "Иран сделал заявление по ядерной сделке и сообщил о готовности продолжать переговоры "
                "по региональной безопасности. В материале приводятся детали переговорного процесса."
            ),
            published_at="2026-04-22T10:00:00",
            source_domain="ria.ru",
            source_name="РИА Новости",
        )
        self._create_article(
            title="Иран сделал заявление",
            body_text="Ваш браузер не поддерживает данный формат видео.",
            published_at="2026-04-22T11:00:00",
            source_domain="ria.ru",
            source_name="РИА Новости",
        )

        result = self.rag_service.answer(
            query="иран сделал заявление",
            date_from="2026-04-01T00:00:00",
            date_to="2026-04-30T23:59:59",
            limit=5,
            source_domains=["ria.ru"],
            include_debug_chunks=True,
        )

        self.assertGreaterEqual(len(result.source_articles), 1)
        self.assertEqual(result.source_articles[0].id, useful_article.id)
        self.assertTrue(all("Ваш браузер не поддерживает" not in chunk.chunk_text for chunk in (result.top_chunks or [])))

    def test_model_reranker_scores_candidates_before_llm_polish(self) -> None:
        target_article = self._create_article(
            title="Тегеран сделал заявление",
            body_text="Тегеран сделал важное заявление по ядерной сделке и региональной безопасности.",
            published_at="2026-04-23T10:00:00",
            source_domain="ria.ru",
            source_name="РИА Новости",
        )
        self._create_article(
            title="Случайная новость",
            body_text="В Москве прошла выставка и открылась новая площадка для мероприятий.",
            published_at="2026-04-23T11:00:00",
            source_domain="ria.ru",
            source_name="РИА Новости",
        )

        result = self.rag_service.answer(
            query="иран",
            date_from="2026-04-01T00:00:00",
            date_to="2026-04-30T23:59:59",
            limit=5,
            include_debug_chunks=True,
            source_domains=["ria.ru"],
        )

        self.assertGreaterEqual(len(self.mock_reranker.calls), 1)
        self.assertGreaterEqual(len(result.source_articles), 1)
        self.assertEqual(result.source_articles[0].id, target_article.id)
        self.assertTrue(any(chunk.model_rerank_score > 0.5 for chunk in (result.top_chunks or [])))

    def test_narrow_query_requires_direct_lexical_evidence(self) -> None:
        target_article = self._create_article(
            title="Block on Telegram discussed in Russia",
            body_text="Telegram may be blocked in Russia. Officials discuss a possible block on the messenger.",
            published_at="2026-04-24T10:00:00",
            source_domain="ria.ru",
            source_name="RIA",
        )
        self._create_article(
            title="Fake YouTube channels reported by Belarusian media",
            body_text="Belarusian media reported fake YouTube channels and broader media platform issues.",
            published_at="2026-04-24T11:00:00",
            source_domain="lenta.ru",
            source_name="Lenta",
        )

        result = self.rag_service.answer(
            query="telegram blocking in russia",
            date_from="2026-04-01T00:00:00",
            date_to="2026-04-30T23:59:59",
            limit=5,
            include_debug_chunks=True,
        )

        self.assertEqual(len(result.source_articles), 1)
        self.assertEqual(result.source_articles[0].id, target_article.id)
        self.assertTrue(
            all(
                "telegram" in f"{chunk.article_title} {chunk.chunk_text}".lower()
                for chunk in (result.top_chunks or [])
            )
        )

    def test_narrow_query_returns_safe_empty_result_when_no_direct_evidence_exists(self) -> None:
        self._create_article(
            title="Fake YouTube channels reported by Belarusian media",
            body_text="Belarusian media reported fake YouTube channels and broader media platform issues.",
            published_at="2026-04-24T11:00:00",
            source_domain="lenta.ru",
            source_name="Lenta",
        )
        self._create_article(
            title="Tax fraud suspects arrested in Moscow",
            body_text="Several suspects in a large tax fraud case were arrested in Moscow.",
            published_at="2026-04-24T12:00:00",
            source_domain="ria.ru",
            source_name="RIA",
        )

        result = self.rag_service.answer(
            query="telegram blocking in russia",
            date_from="2026-04-01T00:00:00",
            date_to="2026-04-30T23:59:59",
            limit=5,
            include_debug_chunks=True,
        )

        self.assertEqual(result.source_articles, [])
        self.assertEqual(result.top_chunks, [])
        self.assertTrue(result.summary_text.strip())


if __name__ == "__main__":
    unittest.main()
