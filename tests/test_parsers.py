from __future__ import annotations

import unittest
from pathlib import Path

from app.ingestion.parsers import LentaParser, RiaParser


class ParserTests(unittest.TestCase):
    def _read_sample(self, filename: str) -> str:
        path = Path("tests") / "samples" / filename
        return path.read_text(encoding="utf-8")

    def test_lenta_parser_extracts_article_fields(self) -> None:
        html = self._read_sample("lenta_article.html")
        parser = LentaParser()

        parsed = parser.parse(html, "https://lenta.ru/news/2026/04/23/example/")

        self.assertEqual(parsed.title, "Лента сообщила о новой инициативе")
        self.assertEqual(parsed.subtitle, "Краткое описание новости для подзаголовка")
        self.assertEqual(parsed.author, "Иван Петров")
        self.assertEqual(parsed.category, "Россия")
        self.assertEqual(parsed.published_at, "2026-04-23T10:15:00+03:00")
        self.assertIn("Первый абзац новости", parsed.body_text)
        self.assertIn("Второй абзац содержит дополнительные детали", parsed.body_text)
        self.assertNotIn("Что думаешь? Оцени!", parsed.body_text)
        self.assertNotIn("Последние новости", parsed.body_text)

    def test_ria_parser_extracts_article_fields(self) -> None:
        html = self._read_sample("ria_article.html")
        parser = RiaParser()

        parsed = parser.parse(html, "https://ria.ru/20260422/test.html")

        self.assertEqual(parsed.title, "РИА: тестовая статья")
        self.assertEqual(parsed.subtitle, "Подзаголовок статьи РИА для теста.")
        self.assertEqual(parsed.author, "Мария Соколова")
        self.assertEqual(parsed.category, "Политика")
        self.assertEqual(parsed.published_at, "2026-04-22T08:30:00+03:00")
        self.assertIn("Первый абзац статьи РИА", parsed.body_text)
        self.assertIn("Второй абзац развивает тему", parsed.body_text)
        self.assertNotIn("Краткий пересказ от РИА ИИ", parsed.body_text)
        self.assertNotIn("Читать ria.ru в", parsed.body_text)


if __name__ == "__main__":
    unittest.main()
