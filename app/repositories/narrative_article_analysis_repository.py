from __future__ import annotations

import sqlite3

from app.models import NarrativeArticleAnalysis
from app.repositories.base import BaseRepository, compact_datetime_sql, normalize_datetime_bound


class NarrativeArticleAnalysisRepository(BaseRepository):
    def upsert_analysis(
        self,
        *,
        article_id: int,
        source_domain: str,
        published_at: str,
        status: str,
        frame_count: int,
        payload_json: str,
    ) -> NarrativeArticleAnalysis:
        self.connection.execute(
            """
            INSERT INTO narrative_article_analyses (
                article_id,
                source_domain,
                published_at,
                status,
                frame_count,
                payload_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(article_id) DO UPDATE SET
                source_domain = excluded.source_domain,
                published_at = excluded.published_at,
                status = excluded.status,
                frame_count = excluded.frame_count,
                payload_json = excluded.payload_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                article_id,
                source_domain,
                normalize_datetime_bound(published_at) or published_at,
                status,
                frame_count,
                payload_json,
            ),
        )
        self.connection.commit()
        record = self.get_by_article_id(article_id)
        if record is None:
            raise RuntimeError("Narrative article analysis could not be loaded after save.")
        return record

    def get_by_article_id(self, article_id: int) -> NarrativeArticleAnalysis | None:
        row = self._fetch_one(
            "SELECT * FROM narrative_article_analyses WHERE article_id = ?",
            (article_id,),
        )
        return self._row_to_record(row) if row else None

    def list_by_date_range_and_sources(
        self,
        *,
        date_from: str,
        date_to: str,
        source_domains: list[str] | None = None,
    ) -> list[NarrativeArticleAnalysis]:
        date_from = normalize_datetime_bound(date_from) or date_from
        date_to = normalize_datetime_bound(date_to) or date_to
        clauses = [f"{compact_datetime_sql('published_at')} BETWEEN ? AND ?"]
        params: list[object] = [date_from, date_to]
        if source_domains:
            placeholders = ", ".join("?" for _ in source_domains)
            clauses.append(f"source_domain IN ({placeholders})")
            params.extend(source_domains)
        rows = self._fetch_all(
            f"""
            SELECT *
            FROM narrative_article_analyses
            WHERE {' AND '.join(clauses)}
            ORDER BY {compact_datetime_sql('published_at')} ASC, article_id ASC
            """,
            tuple(params),
        )
        return [self._row_to_record(row) for row in rows]

    def list_existing_article_ids(
        self,
        *,
        article_ids: list[int],
    ) -> set[int]:
        if not article_ids:
            return set()
        placeholders = ", ".join("?" for _ in article_ids)
        rows = self._fetch_all(
            f"SELECT article_id FROM narrative_article_analyses WHERE article_id IN ({placeholders})",
            tuple(article_ids),
        )
        return {int(row["article_id"]) for row in rows}

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> NarrativeArticleAnalysis:
        return NarrativeArticleAnalysis(
            id=row["id"],
            article_id=row["article_id"],
            source_domain=row["source_domain"],
            published_at=row["published_at"],
            status=row["status"],
            frame_count=row["frame_count"],
            payload_json=row["payload_json"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
