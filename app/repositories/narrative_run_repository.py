from __future__ import annotations

import sqlite3

from app.models.entities import NarrativeRun, NarrativeRunCreate
from app.repositories.base import BaseRepository


class NarrativeRunRepository(BaseRepository):
    def create(self, payload: NarrativeRunCreate) -> NarrativeRun:
        cursor = self.connection.execute(
            """
            INSERT INTO narrative_runs (
                topic_text, date_from, date_to, run_status,
                articles_selected_count, claims_selected_count, finished_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.topic_text,
                payload.date_from,
                payload.date_to,
                payload.run_status,
                payload.articles_selected_count,
                payload.claims_selected_count,
                payload.finished_at,
            ),
        )
        self.connection.commit()
        run = self.get_by_id(cursor.lastrowid)
        if run is None:
            raise RuntimeError("Created narrative run could not be loaded back from the database.")
        return run

    def get_by_id(self, run_id: int) -> NarrativeRun | None:
        row = self._fetch_one("SELECT * FROM narrative_runs WHERE id = ?", (run_id,))
        return self._row_to_run(row) if row else None

    def list(self, limit: int = 100) -> list[NarrativeRun]:
        rows = self._fetch_all(
            """
            SELECT *
            FROM narrative_runs
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [self._row_to_run(row) for row in rows]

    def update_status(
        self,
        run_id: int,
        run_status: str,
        *,
        articles_selected_count: int | None = None,
        claims_selected_count: int | None = None,
        finished_at: str | None = None,
    ) -> bool:
        current = self.get_by_id(run_id)
        if current is None:
            return False

        cursor = self.connection.execute(
            """
            UPDATE narrative_runs
            SET run_status = ?,
                articles_selected_count = ?,
                claims_selected_count = ?,
                finished_at = ?
            WHERE id = ?
            """,
            (
                run_status,
                current.articles_selected_count if articles_selected_count is None else articles_selected_count,
                current.claims_selected_count if claims_selected_count is None else claims_selected_count,
                finished_at,
                run_id,
            ),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> NarrativeRun:
        return NarrativeRun(
            id=row["id"],
            topic_text=row["topic_text"],
            date_from=row["date_from"],
            date_to=row["date_to"],
            run_status=row["run_status"],
            articles_selected_count=row["articles_selected_count"],
            claims_selected_count=row["claims_selected_count"],
            finished_at=row["finished_at"],
            created_at=row["created_at"],
        )
