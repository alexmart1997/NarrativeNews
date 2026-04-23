from __future__ import annotations

import sqlite3

from app.models.entities import Article, ArticleCreate
from app.repositories.base import BaseRepository, bool_to_int


class ArticleRepository(BaseRepository):
    def create_article(self, payload: ArticleCreate) -> Article:
        cursor = self.connection.execute(
            """
            INSERT INTO articles (
                source_id, url, title, subtitle, body_text, published_at, author, category,
                language, content_hash, word_count, is_canonical, duplicate_group_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.source_id,
                payload.url,
                payload.title,
                payload.subtitle,
                payload.body_text,
                payload.published_at,
                payload.author,
                payload.category,
                payload.language,
                payload.content_hash,
                payload.word_count,
                bool_to_int(payload.is_canonical),
                payload.duplicate_group_id,
            ),
        )
        self.connection.commit()
        article = self.get_article_by_id(cursor.lastrowid)
        if article is None:
            raise RuntimeError("Created article could not be loaded back from the database.")
        return article

    def get_article_by_id(self, article_id: int) -> Article | None:
        row = self._fetch_one("SELECT * FROM articles WHERE id = ?", (article_id,))
        return self._row_to_article(row) if row else None

    def get_article_by_url(self, url: str) -> Article | None:
        row = self._fetch_one("SELECT * FROM articles WHERE url = ?", (url,))
        return self._row_to_article(row) if row else None

    def get_article_by_content_hash(self, content_hash: str) -> Article | None:
        row = self._fetch_one(
            """
            SELECT *
            FROM articles
            WHERE content_hash = ?
            ORDER BY is_canonical DESC, id ASC
            LIMIT 1
            """,
            (content_hash,),
        )
        return self._row_to_article(row) if row else None

    def list_articles_by_date_range(self, date_from: str, date_to: str) -> list[Article]:
        rows = self._fetch_all(
            """
            SELECT *
            FROM articles
            WHERE published_at BETWEEN ? AND ?
            ORDER BY published_at ASC, id ASC
            """,
            (date_from, date_to),
        )
        return [self._row_to_article(row) for row in rows]

    def list_canonical_articles_by_date_range(self, date_from: str, date_to: str) -> list[Article]:
        rows = self._fetch_all(
            """
            SELECT *
            FROM articles
            WHERE published_at BETWEEN ? AND ? AND is_canonical = 1
            ORDER BY published_at ASC, id ASC
            """,
            (date_from, date_to),
        )
        return [self._row_to_article(row) for row in rows]

    def mark_article_canonical(
        self,
        article_id: int,
        is_canonical: bool,
        duplicate_group_id: str | None = None,
    ) -> bool:
        cursor = self.connection.execute(
            """
            UPDATE articles
            SET is_canonical = ?, duplicate_group_id = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (bool_to_int(is_canonical), duplicate_group_id, article_id),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def list_by_source(self, source_id: int, limit: int = 100) -> list[Article]:
        rows = self._fetch_all(
            """
            SELECT *
            FROM articles
            WHERE source_id = ?
            ORDER BY published_at DESC, id DESC
            LIMIT ?
            """,
            (source_id, limit),
        )
        return [self._row_to_article(row) for row in rows]

    def create_duplicate_record(
        self,
        *,
        duplicate_group_id: str,
        article_id: int,
        duplicate_type: str,
        is_primary: bool,
        similarity_score: float | None = None,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO article_duplicates (
                duplicate_group_id, article_id, duplicate_type, is_primary, similarity_score
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                duplicate_group_id,
                article_id,
                duplicate_type,
                bool_to_int(is_primary),
                similarity_score,
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    @staticmethod
    def _row_to_article(row: sqlite3.Row) -> Article:
        return Article(
            id=row["id"],
            source_id=row["source_id"],
            url=row["url"],
            title=row["title"],
            subtitle=row["subtitle"],
            body_text=row["body_text"],
            published_at=row["published_at"],
            author=row["author"],
            category=row["category"],
            language=row["language"],
            content_hash=row["content_hash"],
            word_count=row["word_count"],
            is_canonical=bool(row["is_canonical"]),
            duplicate_group_id=row["duplicate_group_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
