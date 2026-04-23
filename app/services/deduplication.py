from __future__ import annotations

from dataclasses import dataclass

from app.repositories import ArticleRepository
from app.services.normalization import NormalizedArticle


@dataclass(frozen=True, slots=True)
class DeduplicationResult:
    is_duplicate: bool
    duplicate_type: str | None
    canonical_article_id: int | None
    duplicate_group_id: str | None


class DeduplicationService:
    def __init__(self, article_repository: ArticleRepository) -> None:
        self.article_repository = article_repository

    def check_duplicate(self, article: NormalizedArticle) -> DeduplicationResult:
        existing_by_url = self.article_repository.get_article_by_url(article.url)
        if existing_by_url is not None:
            return DeduplicationResult(
                is_duplicate=True,
                duplicate_type="url",
                canonical_article_id=existing_by_url.id,
                duplicate_group_id=existing_by_url.duplicate_group_id or f"url:{existing_by_url.id}",
            )

        existing_by_hash = self.article_repository.get_article_by_content_hash(article.content_hash)
        if existing_by_hash is not None:
            return DeduplicationResult(
                is_duplicate=True,
                duplicate_type="content_hash",
                canonical_article_id=existing_by_hash.id,
                duplicate_group_id=existing_by_hash.duplicate_group_id or f"hash:{existing_by_hash.content_hash}",
            )

        return DeduplicationResult(
            is_duplicate=False,
            duplicate_type=None,
            canonical_article_id=None,
            duplicate_group_id=None,
        )
