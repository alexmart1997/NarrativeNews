from __future__ import annotations

import sqlite3

from app.models import Article, NarrativeResult, NarrativeResultArticleCreate, NarrativeResultCreate
from app.repositories.base import BaseRepository


class NarrativeResultRepository(BaseRepository):
    def create(self, payload: NarrativeResultCreate) -> NarrativeResult:
        cursor = self.connection.execute(
            """
            INSERT INTO narrative_results (run_id, narrative_type, title, formulation, explanation, strength_score)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                payload.run_id,
                payload.narrative_type,
                payload.title,
                payload.formulation,
                payload.explanation,
                payload.strength_score,
            ),
        )
        self.connection.commit()
        row = self._fetch_one("SELECT * FROM narrative_results WHERE id = ?", (cursor.lastrowid,))
        if row is None:
            raise RuntimeError("Created narrative result could not be loaded back from the database.")
        return self._row_to_result(row)

    def create_result_articles(self, payloads: list[NarrativeResultArticleCreate]) -> None:
        for payload in payloads:
            self.connection.execute(
                """
                INSERT INTO narrative_result_articles (narrative_result_id, article_id, rank, selection_reason)
                VALUES (?, ?, ?, ?)
                """,
                (
                    payload.narrative_result_id,
                    payload.article_id,
                    payload.rank,
                    payload.selection_reason,
                ),
            )
        self.connection.commit()

    def list_by_run_id(self, run_id: int) -> list[NarrativeResult]:
        rows = self._fetch_all(
            "SELECT * FROM narrative_results WHERE run_id = ? ORDER BY strength_score DESC, id ASC",
            (run_id,),
        )
        return [self._row_to_result(row) for row in rows]

    def list_articles_for_result(self, narrative_result_id: int) -> list[Article]:
        rows = self._fetch_all(
            """
            SELECT a.*
            FROM narrative_result_articles nra
            INNER JOIN articles a ON a.id = nra.article_id
            WHERE nra.narrative_result_id = ?
            ORDER BY nra.rank ASC, a.id ASC
            """,
            (narrative_result_id,),
        )
        return [self._row_to_article(row) for row in rows]

    @staticmethod
    def _row_to_result(row: sqlite3.Row) -> NarrativeResult:
        return NarrativeResult(
            id=row["id"],
            run_id=row["run_id"],
            narrative_type=row["narrative_type"],
            title=row["title"],
            formulation=row["formulation"],
            explanation=row["explanation"],
            strength_score=row["strength_score"],
            created_at=row["created_at"],
        )

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
