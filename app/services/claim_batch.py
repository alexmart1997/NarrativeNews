from __future__ import annotations

from dataclasses import dataclass

from app.repositories import ArticleRepository, ClaimRepository
from app.services.claim_extraction import ClaimExtractor


@dataclass(frozen=True, slots=True)
class ClaimBatchResult:
    processed_articles: int
    articles_with_claims: int
    claims_created: int
    skipped_articles: int
    failed_articles: int


class ClaimBatchService:
    def __init__(
        self,
        *,
        article_repository: ArticleRepository,
        claim_repository: ClaimRepository,
        claim_extractor: ClaimExtractor,
    ) -> None:
        self.article_repository = article_repository
        self.claim_repository = claim_repository
        self.claim_extractor = claim_extractor

    def extract_missing_claims(
        self,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 100,
    ) -> ClaimBatchResult:
        articles = self.article_repository.list_canonical_articles_without_claims(
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )
        processed_articles = 0
        articles_with_claims = 0
        claims_created = 0
        skipped_articles = 0
        failed_articles = 0

        for article in articles:
            processed_articles += 1
            try:
                claims = self.claim_extractor.extract(article)
                if not claims:
                    skipped_articles += 1
                    continue
                created = self.claim_repository.create_many(claims)
                articles_with_claims += 1
                claims_created += len(created)
            except Exception:
                failed_articles += 1

        return ClaimBatchResult(
            processed_articles=processed_articles,
            articles_with_claims=articles_with_claims,
            claims_created=claims_created,
            skipped_articles=skipped_articles,
            failed_articles=failed_articles,
        )
