from __future__ import annotations

from app.ingestion.parsers.base import ParsedArticle


class ParsedArticleValidationError(ValueError):
    """Raised when a parsed article does not meet minimum quality thresholds."""


def validate_parsed_article(parsed_article: ParsedArticle, *, min_body_length: int = 120) -> None:
    if not parsed_article.title.strip():
        raise ParsedArticleValidationError("Parsed article title is empty.")
    if not parsed_article.body_text.strip():
        raise ParsedArticleValidationError("Parsed article body is empty.")
    if len(parsed_article.body_text.strip()) < min_body_length:
        raise ParsedArticleValidationError(
            f"Parsed article body is shorter than minimum threshold: {min_body_length}"
        )
    if not parsed_article.published_at.strip():
        raise ParsedArticleValidationError("Parsed article published_at is empty.")


def validate_normalized_article(*, title: str, body_text: str) -> None:
    if not title.strip():
        raise ParsedArticleValidationError("Normalized article title is empty.")
    if not body_text.strip():
        raise ParsedArticleValidationError("Normalized article body is empty.")
