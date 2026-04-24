from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from app.db.connection import create_connection
from app.db.schema import create_schema
from app.models import ArticleCreate, SourceCreate
from app.repositories import ArticleRepository, ClaimRepository, SourceRepository
from app.services import BaseLLMClient, ClaimBatchService, ClaimExtractor


class MockClaimLLMClient(BaseLLMClient):
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[str] = []

    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        raise AssertionError("generate_text should not be called directly in this test")

    def generate_json(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, object]:
        self.calls.append(prompt)
        return self.payload


class ClaimBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path("tests") / ".tmp" / f"claim-batch-{uuid.uuid4().hex}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.temp_dir / "claim-batch.db"
        self.connection = create_connection(self.database_path)
        create_schema(self.connection)
        self.source_repo = SourceRepository(self.connection)
        self.article_repo = ArticleRepository(self.connection)
        self.claim_repo = ClaimRepository(self.connection)
        self.source = self.source_repo.create(
            SourceCreate(
                name="Example",
                domain=f"claim-batch-{uuid.uuid4().hex[:8]}.ru",
                base_url="https://example.ru",
                source_type="news_site",
            )
        )

    def tearDown(self) -> None:
        self.connection.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_article(self, *, url_suffix: str, published_at: str) -> int:
        article = self.article_repo.create_article(
            ArticleCreate(
                source_id=self.source.id,
                url=f"https://example.ru/articles/{url_suffix}",
                title="Тестовая статья",
                subtitle=None,
                body_text=(
                    "Министр заявил, что поставки вырастут к осени. "
                    "Это должно снизить дефицит на рынке.\n\n"
                    "Аналитики считают, что меры приведут к снижению цен."
                ),
                published_at=published_at,
                content_hash=uuid.uuid4().hex,
                word_count=20,
                is_canonical=True,
            )
        )
        return article.id

    def test_batch_extracts_claims_for_missing_articles(self) -> None:
        self._create_article(url_suffix="one", published_at="2026-04-20T10:00:00")
        self._create_article(url_suffix="two", published_at="2026-04-21T10:00:00")
        extractor = ClaimExtractor(
            llm_client=MockClaimLLMClient(
                {
                    "claims": [
                        {
                            "claim_text": "Министр заявил, что поставки вырастут к осени.",
                            "normalized_claim_text": "Поставки вырастут к осени",
                            "claim_type": "predictive",
                            "extraction_confidence": 0.9,
                            "classification_confidence": 0.88,
                            "source_sentence": "Министр заявил, что поставки вырастут к осени.",
                            "source_paragraph_index": 0,
                        }
                    ]
                }
            )
        )
        service = ClaimBatchService(
            article_repository=self.article_repo,
            claim_repository=self.claim_repo,
            claim_extractor=extractor,
        )

        result = service.extract_missing_claims(
            date_from="2026-04-20T00:00:00",
            date_to="2026-04-21T23:59:59",
            limit=10,
        )

        self.assertEqual(result.processed_articles, 2)
        self.assertEqual(result.articles_with_claims, 2)
        self.assertEqual(result.claims_created, 2)

    def test_batch_skips_articles_that_already_have_claims(self) -> None:
        first_id = self._create_article(url_suffix="one", published_at="2026-04-20T10:00:00")
        self._create_article(url_suffix="two", published_at="2026-04-21T10:00:00")
        existing_article = self.article_repo.get_article_by_id(first_id)
        assert existing_article is not None
        self.claim_repo.create_many(
            ClaimExtractor(
                llm_client=MockClaimLLMClient(
                    {
                        "claims": [
                            {
                                "claim_text": "Министр заявил, что поставки вырастут к осени.",
                                "normalized_claim_text": "Поставки вырастут к осени",
                                "claim_type": "predictive",
                            }
                        ]
                    }
                )
            ).extract(existing_article)
        )
        extractor = ClaimExtractor(
            llm_client=MockClaimLLMClient(
                {
                    "claims": [
                        {
                            "claim_text": "Аналитики считают, что меры приведут к снижению цен.",
                            "normalized_claim_text": "Меры приведут к снижению цен",
                            "claim_type": "causal",
                        }
                    ]
                }
            )
        )
        service = ClaimBatchService(
            article_repository=self.article_repo,
            claim_repository=self.claim_repo,
            claim_extractor=extractor,
        )

        result = service.extract_missing_claims(limit=10)

        self.assertEqual(result.processed_articles, 1)
        self.assertEqual(result.articles_with_claims, 1)
        self.assertEqual(len(self.claim_repo.list_by_type("causal")), 1)


if __name__ == "__main__":
    unittest.main()
