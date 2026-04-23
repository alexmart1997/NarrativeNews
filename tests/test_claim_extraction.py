from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from app.db.connection import create_connection
from app.db.schema import create_schema
from app.models import ArticleCreate, ClaimDraft, SentenceContext, SourceCreate
from app.repositories import ArticleRepository, ClaimRepository, SourceRepository
from app.services import BaseClaimLLMClient, ClaimExtractor


class MockClaimLLMClient(BaseClaimLLMClient):
    def __init__(self, drafts: list[ClaimDraft]) -> None:
        self.drafts = drafts
        self.calls: list[tuple[int, list[SentenceContext]]] = []

    def extract_claims(self, article, sentences: list[SentenceContext]) -> list[ClaimDraft]:
        self.calls.append((article.id, sentences))
        return self.drafts


class ClaimExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path("tests") / ".tmp" / f"claims-{uuid.uuid4().hex}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.temp_dir / "claims.db"
        self.connection = create_connection(self.database_path)
        create_schema(self.connection)
        self.source_repo = SourceRepository(self.connection)
        self.article_repo = ArticleRepository(self.connection)
        self.claim_repo = ClaimRepository(self.connection)

    def tearDown(self) -> None:
        self.connection.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_article(self, *, is_canonical: bool = True):
        source = self.source_repo.create(
            SourceCreate(
                name="Example",
                domain=f"claims-{uuid.uuid4().hex[:8]}.ru",
                base_url="https://example.ru",
                source_type="news_site",
            )
        )
        return self.article_repo.create_article(
            ArticleCreate(
                source_id=source.id,
                url=f"https://example.ru/articles/{uuid.uuid4().hex}",
                title="Тестовая статья",
                subtitle=None,
                body_text=(
                    "Министр заявил, что поставки вырастут к осени. "
                    "Это должно снизить дефицит на рынке.\n\n"
                    "Поделиться новостью. "
                    "Аналитики считают, что меры приведут к снижению цен."
                ),
                published_at="2026-04-23T10:00:00",
                content_hash=uuid.uuid4().hex,
                word_count=24,
                is_canonical=is_canonical,
            )
        )

    def test_extraction_from_single_article(self) -> None:
        article = self._create_article()
        mock_llm = MockClaimLLMClient(
            drafts=[
                ClaimDraft(
                    claim_text="Министр заявил, что поставки вырастут к осени.",
                    normalized_claim_text="Поставки вырастут к осени",
                    claim_type="predictive",
                    extraction_confidence=0.9,
                    classification_confidence=0.88,
                    source_sentence="Министр заявил, что поставки вырастут к осени.",
                    source_paragraph_index=0,
                ),
                ClaimDraft(
                    claim_text="Поделиться новостью.",
                    normalized_claim_text="Поделиться новостью",
                    claim_type="other",
                    extraction_confidence=0.2,
                    classification_confidence=0.2,
                    source_sentence="Поделиться новостью.",
                    source_paragraph_index=1,
                ),
            ]
        )
        extractor = ClaimExtractor(llm_client=mock_llm)

        claims = extractor.extract(article)

        self.assertEqual(len(mock_llm.calls), 1)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].claim_type, "predictive")
        self.assertEqual(claims[0].normalized_claim_text, "Поставки вырастут к осени")

    def test_claims_saved_in_database(self) -> None:
        article = self._create_article()
        extractor = ClaimExtractor(
            llm_client=MockClaimLLMClient(
                drafts=[
                    ClaimDraft(
                        claim_text="Это должно снизить дефицит на рынке.",
                        normalized_claim_text="Снижение дефицита на рынке",
                        claim_type="causal",
                        extraction_confidence=0.85,
                        classification_confidence=0.83,
                        source_sentence="Это должно снизить дефицит на рынке.",
                        source_paragraph_index=0,
                    )
                ]
            )
        )

        created = self.claim_repo.create_many(extractor.extract(article))
        loaded = self.claim_repo.list_by_article_id(article.id)

        self.assertEqual(len(created), 1)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].claim_type, "causal")

    def test_noncanonical_article_produces_no_claims(self) -> None:
        article = self._create_article(is_canonical=False)
        extractor = ClaimExtractor(
            llm_client=MockClaimLLMClient(
                drafts=[
                    ClaimDraft(
                        claim_text="Аналитики считают, что меры приведут к снижению цен.",
                        normalized_claim_text="Меры приведут к снижению цен",
                        claim_type="causal",
                    )
                ]
            )
        )

        claims = extractor.extract(article)

        self.assertEqual(claims, [])


if __name__ == "__main__":
    unittest.main()
