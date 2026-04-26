from __future__ import annotations

import json
import sqlite3

from app.models import NarrativeAnalysisRun
from app.repositories.base import BaseRepository, normalize_datetime_bound


class NarrativeAnalysisRepository(BaseRepository):
    def save_run(
        self,
        *,
        source_domains_key: str,
        date_from: str,
        date_to: str,
        payload_json: str,
        status: str,
        documents_count: int,
        topics_count: int,
        frames_count: int,
        clusters_count: int,
        labels_count: int,
        assignments_count: int,
        dynamics_count: int,
    ) -> NarrativeAnalysisRun:
        cursor = self.connection.execute(
            """
            INSERT INTO narrative_analysis_runs (
                source_domains_key,
                date_from,
                date_to,
                status,
                documents_count,
                topics_count,
                frames_count,
                clusters_count,
                labels_count,
                assignments_count,
                dynamics_count,
                payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_domains_key,
                normalize_datetime_bound(date_from) or date_from,
                normalize_datetime_bound(date_to) or date_to,
                status,
                documents_count,
                topics_count,
                frames_count,
                clusters_count,
                labels_count,
                assignments_count,
                dynamics_count,
                payload_json,
            ),
        )
        self.connection.commit()
        run = self.get_by_id(int(cursor.lastrowid))
        if run is None:
            raise RuntimeError("Narrative analysis run could not be loaded after save.")
        return run

    def get_by_id(self, run_id: int) -> NarrativeAnalysisRun | None:
        row = self._fetch_one("SELECT * FROM narrative_analysis_runs WHERE id = ?", (run_id,))
        return self._row_to_run(row) if row else None

    def get_latest_run(
        self,
        *,
        source_domains_key: str,
        date_from: str,
        date_to: str,
    ) -> NarrativeAnalysisRun | None:
        row = self._fetch_one(
            """
            SELECT *
            FROM narrative_analysis_runs
            WHERE source_domains_key = ? AND date_from = ? AND date_to = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (
                source_domains_key,
                normalize_datetime_bound(date_from) or date_from,
                normalize_datetime_bound(date_to) or date_to,
            ),
        )
        return self._row_to_run(row) if row else None

    def get_latest_payload(
        self,
        *,
        source_domains_key: str,
        date_from: str,
        date_to: str,
    ) -> dict[str, object] | None:
        run = self.get_latest_run(
            source_domains_key=source_domains_key,
            date_from=date_from,
            date_to=date_to,
        )
        if run is None:
            return None
        return json.loads(run.payload_json)

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> NarrativeAnalysisRun:
        return NarrativeAnalysisRun(
            id=row["id"],
            source_domains_key=row["source_domains_key"],
            date_from=row["date_from"],
            date_to=row["date_to"],
            status=row["status"],
            documents_count=row["documents_count"],
            topics_count=row["topics_count"],
            frames_count=row["frames_count"],
            clusters_count=row["clusters_count"],
            labels_count=row["labels_count"],
            assignments_count=row["assignments_count"],
            dynamics_count=row["dynamics_count"],
            payload_json=row["payload_json"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
