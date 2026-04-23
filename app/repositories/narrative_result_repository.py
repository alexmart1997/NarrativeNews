from __future__ import annotations

import sqlite3

from app.models import NarrativeResult, NarrativeResultArticleCreate, NarrativeResultCreate
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
