from __future__ import annotations

import hashlib
import unittest

from app.ingestion.parsers import ParsedArticle
from app.services import ArticleNormalizer


class NormalizationTests(unittest.TestCase):
    def test_normalization_cleans_fields_and_computes_hash(self) -> None:
        normalizer = ArticleNormalizer()
        parsed_article = ParsedArticle(
            title="  Заголовок   статьи  ",
            subtitle="  Подзаголовок  ",
            body_text=" Первая строка текста.\n\n\n  Вторая строка текста. ",
            published_at="2026-04-23T10:00:00+03:00",
            author="  Автор  ",
            category="  Политика  ",
        )

        normalized = normalizer.normalize(
            parsed_article,
            url="HTTPS://RIA.RU/world/20260423/test/?utm_source=x&utm_medium=email&id=5#fragment",
        )

        self.assertEqual(normalized.title, "Заголовок статьи")
        self.assertEqual(normalized.subtitle, "Подзаголовок")
        self.assertEqual(normalized.body_text, "Первая строка текста.\n\nВторая строка текста.")
        self.assertEqual(normalized.url, "https://ria.ru/world/20260423/test?id=5")
        self.assertEqual(normalized.word_count, 6)
        self.assertEqual(
            normalized.content_hash,
            hashlib.sha256(normalized.body_text.encode("utf-8")).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
