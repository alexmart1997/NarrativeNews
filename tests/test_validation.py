from __future__ import annotations

import unittest

from app.ingestion.parsers import ParsedArticle
from app.ingestion.pipeline import ParsedArticleValidationError, validate_parsed_article


class ValidationTests(unittest.TestCase):
    def test_validation_accepts_well_formed_article(self) -> None:
        article = ParsedArticle(
            title="Заголовок",
            subtitle=None,
            body_text="Это достаточно длинный текст статьи для прохождения проверки валидатора и сохранения в базу.",
            published_at="2026-04-23T10:00:00+03:00",
            author=None,
            category=None,
        )
        validate_parsed_article(article, min_body_length=50)

    def test_validation_rejects_short_body(self) -> None:
        article = ParsedArticle(
            title="Заголовок",
            subtitle=None,
            body_text="Слишком коротко.",
            published_at="2026-04-23T10:00:00+03:00",
            author=None,
            category=None,
        )
        with self.assertRaises(ParsedArticleValidationError):
            validate_parsed_article(article, min_body_length=50)

    def test_validation_rejects_missing_title(self) -> None:
        article = ParsedArticle(
            title="",
            subtitle=None,
            body_text="Это достаточно длинный текст статьи для прохождения проверки валидатора и сохранения в базу.",
            published_at="2026-04-23T10:00:00+03:00",
            author=None,
            category=None,
        )
        with self.assertRaises(ParsedArticleValidationError):
            validate_parsed_article(article, min_body_length=50)


if __name__ == "__main__":
    unittest.main()
