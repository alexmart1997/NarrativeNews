from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from app.db.connection import create_connection
from app.db.schema import create_schema
from app.models import ArticleCreate, ClaimCreate, SourceCreate
from app.repositories import (
    ArticleRepository,
    ClaimClusterRepository,
    ClaimRepository,
    NarrativeResultRepository,
    NarrativeRunRepository,
    SourceRepository,
)
from app.services import (
    BaseEmbeddingClient,
    BaseLLMClient,
    ClaimGrouper,
    NarrativeLabelingService,
    NarrativeRunService,
    NarrativeScorer,
)


class MockNarrativeRunLabelingLLMClient(BaseLLMClient):
    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        raise AssertionError("generate_text should not be called in this test")

    def generate_json(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, object]:
        if "predictive" in prompt:
            return {
                "title": "predictive label",
                "formulation": "predictive formulation",
                "explanation": "predictive explanation",
            }
        if "causal" in prompt:
            return {
                "title": "causal label",
                "formulation": "causal formulation",
                "explanation": "causal explanation",
            }
        return {
            "title": "meta label",
            "formulation": "meta formulation",
            "explanation": "meta explanation",
        }


class MockClaimEmbeddingClient(BaseEmbeddingClient):
    @property
    def model_name(self) -> str:
        return "mock-claims"

    def embed_text(self, text: str) -> list[float]:
        lowered = text.lower()
        if "инфляц" in lowered or "цен" in lowered:
            return [1.0, 0.0]
        if "росси" in lowered:
            return [0.7, 0.0]
        if "спрос" in lowered or "причин" in lowered:
            return [0.0, 1.0]
        return [0.3, 0.3]


class NarrativeRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path("tests") / ".tmp" / f"narrative-{uuid.uuid4().hex}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.temp_dir / "narrative.db"
        self.connection = create_connection(self.database_path)
        create_schema(self.connection)
        self.source_repo = SourceRepository(self.connection)
        self.article_repo = ArticleRepository(self.connection)
        self.claim_repo = ClaimRepository(self.connection)
        self.run_repo = NarrativeRunRepository(self.connection)
        self.cluster_repo = ClaimClusterRepository(self.connection)
        self.result_repo = NarrativeResultRepository(self.connection)
        self.embedding_client = MockClaimEmbeddingClient()
        self.service = NarrativeRunService(
            article_repository=self.article_repo,
            claim_repository=self.claim_repo,
            narrative_run_repository=self.run_repo,
            claim_cluster_repository=self.cluster_repo,
            narrative_result_repository=self.result_repo,
            claim_grouper=ClaimGrouper(embedding_client=self.embedding_client),
            narrative_scorer=NarrativeScorer(),
            narrative_labeling_service=NarrativeLabelingService(
                llm_client=MockNarrativeRunLabelingLLMClient()
            ),
            embedding_client=self.embedding_client,
        )

    def tearDown(self) -> None:
        self.connection.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _seed_article_with_claims(
        self,
        *,
        title: str,
        body_text: str,
        published_at: str,
        claims: list[tuple[str, str, str]],
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
                is_canonical=True,
            )
        )
        self.claim_repo.create_many(
            [
                ClaimCreate(
                    article_id=article.id,
                    claim_text=claim_text,
                    normalized_claim_text=normalized_claim_text,
                    claim_type=claim_type,
                    extraction_confidence=0.9,
                    classification_confidence=0.85,
                    source_sentence=claim_text,
                    source_paragraph_index=0,
                )
                for claim_text, normalized_claim_text, claim_type in claims
            ]
        )
        return article

    def test_grouping_claims_merges_close_normalized_text(self) -> None:
        article = self._seed_article_with_claims(
            title="Инфляция",
            body_text="Инфляция растет.",
            published_at="2026-04-20T10:00:00",
            claims=[
                ("Инфляция вырастет к лету.", "инфляция вырастет к лету", "predictive"),
                ("Рост цен ожидается летом.", "рост цен ожидается летом", "predictive"),
            ],
        )

        claims = self.claim_repo.list_by_article_id(article.id)
        grouped = ClaimGrouper(embedding_client=self.embedding_client).group(claims, {article.id: article})

        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0].claim_type, "predictive")
        self.assertEqual(len(grouped[0].claims), 2)
        self.assertTrue(grouped[0].cluster_summary)

    def test_narrative_run_creates_run_and_results(self) -> None:
        self._seed_article_with_claims(
            title="Инфляция и рынок",
            body_text="Инфляция влияет на рынок.",
            published_at="2026-04-20T10:00:00",
            claims=[
                ("Инфляция вырастет к лету.", "инфляция вырастет к лету", "predictive"),
                ("Рост цен вызвал пересмотр планов компаний.", "рост цен вызвал пересмотр планов компаний", "causal"),
                ("ЦБ сообщил об ускорении инфляции.", "цб сообщил об ускорении инфляции", "meta"),
                ("Ранее сообщалось о рынке.", "ранее сообщалось о рынке", "other"),
            ],
        )
        self._seed_article_with_claims(
            title="Инфляция и ожидания",
            body_text="Ожидания инфляции меняются.",
            published_at="2026-04-21T10:00:00",
            claims=[
                ("Рост цен ожидается летом.", "рост цен ожидается летом", "predictive"),
                ("Спрос стал причиной пересмотра цен.", "спрос стал причиной пересмотра цен", "causal"),
            ],
        )

        result = self.service.run(
            topic_text="инфляция",
            date_from="2026-04-01T00:00:00",
            date_to="2026-04-30T23:59:59",
        )

        run = result["run"]
        clusters = result["clusters"]
        narrative_results = result["results"]

        self.assertIsNotNone(run)
        self.assertEqual(run.run_status, "completed")
        self.assertEqual(run.articles_selected_count, 2)
        self.assertEqual(run.claims_selected_count, 5)
        self.assertGreaterEqual(len(clusters), 3)
        self.assertGreaterEqual(len(narrative_results), 3)

    def test_top_one_cluster_per_type_is_saved(self) -> None:
        self._seed_article_with_claims(
            title="Инфляция 1",
            body_text="Инфляция продолжится.",
            published_at="2026-04-20T10:00:00",
            claims=[
                ("Инфляция вырастет к лету.", "инфляция вырастет к лету", "predictive"),
                ("Рост цен ожидается летом.", "рост цен ожидается летом", "predictive"),
                ("Рост цен вызвал коррекцию спроса.", "рост цен вызвал коррекцию спроса", "causal"),
                ("ЦБ сообщил об ускорении инфляции.", "цб сообщил об ускорении инфляции", "meta"),
            ],
        )
        self._seed_article_with_claims(
            title="Инфляция 2",
            body_text="Инфляция продолжается.",
            published_at="2026-04-21T10:00:00",
            claims=[
                ("Инфляция вырастет к лету.", "инфляция вырастет к лету", "predictive"),
                ("Спрос стал причиной пересмотра цен.", "спрос стал причиной пересмотра цен", "causal"),
                ("Минфин сообщил об ускорении инфляции.", "минфин сообщил об ускорении инфляции", "meta"),
            ],
        )

        result = self.service.run(
            topic_text="инфляция",
            date_from="2026-04-01T00:00:00",
            date_to="2026-04-30T23:59:59",
        )

        run = result["run"]
        saved_results = self.result_repo.list_by_run_id(run.id)

        self.assertEqual({item.narrative_type for item in saved_results}, {"predictive", "causal", "meta"})
        predictive = next(item for item in saved_results if item.narrative_type == "predictive")
        self.assertEqual(predictive.title, "predictive label")

    def test_global_narrative_run_works_without_topic(self) -> None:
        self._seed_article_with_claims(
            title="Экономика",
            body_text="В корпусе есть прогнозы и причины.",
            published_at="2026-04-10T10:00:00",
            claims=[
                ("Инфляция вырастет к лету.", "инфляция вырастет к лету", "predictive"),
                ("Рост цен вызвал пересмотр спроса.", "рост цен вызвал пересмотр спроса", "causal"),
                ("ЦБ сообщил об ускорении инфляции.", "цб сообщил об ускорении инфляции", "meta"),
            ],
        )
        self._seed_article_with_claims(
            title="Рынок",
            body_text="В корпусе есть еще один поддерживающий набор claims.",
            published_at="2026-04-11T10:00:00",
            claims=[
                ("Рост цен ожидается летом.", "рост цен ожидается летом", "predictive"),
                ("Спрос стал причиной пересмотра цен.", "спрос стал причиной пересмотра цен", "causal"),
            ],
        )

        result = self.service.run(
            topic_text=None,
            date_from="2026-04-01T00:00:00",
            date_to="2026-04-30T23:59:59",
        )

        run = result["run"]
        narrative_results = result["results"]

        self.assertEqual(run.topic_text, "")
        self.assertEqual(run.run_status, "completed")
        self.assertEqual(run.articles_selected_count, 2)
        self.assertEqual(run.claims_selected_count, 5)
        self.assertEqual({item.narrative_type for item in narrative_results}, {"predictive", "causal", "meta"})

    def test_global_mode_suppresses_banal_meta_narratives(self) -> None:
        fallback_service = NarrativeRunService(
            article_repository=self.article_repo,
            claim_repository=self.claim_repo,
            narrative_run_repository=self.run_repo,
            claim_cluster_repository=self.cluster_repo,
            narrative_result_repository=self.result_repo,
            claim_grouper=ClaimGrouper(embedding_client=self.embedding_client),
            narrative_scorer=NarrativeScorer(),
            narrative_labeling_service=NarrativeLabelingService(),
            embedding_client=self.embedding_client,
        )
        self._seed_article_with_claims(
            title="Шаблонный meta",
            body_text="Короткий формальный комментарий.",
            published_at="2026-04-12T10:00:00",
            claims=[
                ("Министр сообщил о заседании.", "министр сообщил о заседании", "meta"),
            ],
        )
        self._seed_article_with_claims(
            title="Содержательный meta",
            body_text="Есть содержательное сообщение с фактом.",
            published_at="2026-04-13T10:00:00",
            claims=[
                ("ЦБ сообщил об ускорении инфляции до 9 процентов.", "цб сообщил об ускорении инфляции до 9 процентов", "meta"),
            ],
        )

        result = fallback_service.run(
            topic_text=None,
            date_from="2026-04-01T00:00:00",
            date_to="2026-04-30T23:59:59",
        )

        meta_result = next(item for item in result["results"] if item.narrative_type == "meta")
        self.assertIn("ускорении инфляции", meta_result.formulation.lower())

    def test_narrative_run_can_filter_by_source_domain(self) -> None:
        self._seed_article_with_claims(
            title="РИА экономика",
            body_text="Инфляция и спрос в РИА.",
            published_at="2026-04-14T10:00:00",
            source_domain="ria.ru",
            source_name="РИА Новости",
            claims=[
                ("Инфляция вырастет к лету.", "инфляция вырастет к лету", "predictive"),
                ("Рост цен вызвал пересмотр спроса.", "рост цен вызвал пересмотр спроса", "causal"),
                ("ЦБ сообщил об ускорении инфляции.", "цб сообщил об ускорении инфляции", "meta"),
            ],
        )
        self._seed_article_with_claims(
            title="Лента экономика",
            body_text="Инфляция и спрос в Ленте.",
            published_at="2026-04-15T10:00:00",
            source_domain="lenta.ru",
            source_name="Лента.ру",
            claims=[
                ("Инфляция замедлится осенью.", "инфляция замедлится осенью", "predictive"),
                ("Снижение спроса вызвало пересмотр цен.", "снижение спроса вызвало пересмотр цен", "causal"),
                ("Минфин сообщил о пересмотре прогноза.", "минфин сообщил о пересмотре прогноза", "meta"),
            ],
        )

        result = self.service.run(
            topic_text=None,
            date_from="2026-04-01T00:00:00",
            date_to="2026-04-30T23:59:59",
            source_domains=["ria.ru"],
        )

        self.assertEqual(result["run"].articles_selected_count, 1)
        self.assertEqual(len(result["articles"]), 1)
        self.assertTrue(all("ria.ru" in article.url for article in result["articles"]))

    def test_global_narrative_run_supports_iso_date_filters_for_compact_published_at(self) -> None:
        self._seed_article_with_claims(
            title="РИА экономика",
            body_text="Инфляция и спрос в РИА.",
            published_at="20260414T1010",
            source_domain="ria.ru",
            source_name="РИА Новости",
            claims=[
                ("Инфляция вырастет к лету.", "инфляция вырастет к лету", "predictive"),
                ("Рост цен вызвал пересмотр спроса.", "рост цен вызвал пересмотр спроса", "causal"),
                ("ЦБ сообщил об ускорении инфляции до 9 процентов.", "цб сообщил об ускорении инфляции до 9 процентов", "meta"),
            ],
        )

        result = self.service.run(
            topic_text=None,
            date_from="2026-04-01T00:00:00",
            date_to="2026-04-30T23:59:59",
            source_domains=["ria.ru"],
        )

        self.assertEqual(result["run"].articles_selected_count, 1)
        self.assertEqual({item.narrative_type for item in result["results"]}, {"predictive", "causal", "meta"})

if __name__ == "__main__":
    unittest.main()
