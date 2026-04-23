from __future__ import annotations

import sqlite3

from app.models import ClaimCluster, ClaimClusterCreate, ClaimClusterItemCreate
from app.repositories.base import BaseRepository, bool_to_int


class ClaimClusterRepository(BaseRepository):
    def create(self, payload: ClaimClusterCreate) -> ClaimCluster:
        cursor = self.connection.execute(
            """
            INSERT INTO claim_clusters (
                run_id, claim_type, cluster_label, cluster_summary,
                cluster_score, claim_count, article_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.run_id,
                payload.claim_type,
                payload.cluster_label,
                payload.cluster_summary,
                payload.cluster_score,
                payload.claim_count,
                payload.article_count,
            ),
        )
        self.connection.commit()
        row = self._fetch_one("SELECT * FROM claim_clusters WHERE id = ?", (cursor.lastrowid,))
        if row is None:
            raise RuntimeError("Created cluster could not be loaded back from the database.")
        return self._row_to_cluster(row)

    def create_items(self, payloads: list[ClaimClusterItemCreate]) -> None:
        for payload in payloads:
            self.connection.execute(
                """
                INSERT INTO claim_cluster_items (cluster_id, claim_id, membership_score, is_representative)
                VALUES (?, ?, ?, ?)
                """,
                (
                    payload.cluster_id,
                    payload.claim_id,
                    payload.membership_score,
                    bool_to_int(payload.is_representative),
                ),
            )
        self.connection.commit()

    def list_by_run_id(self, run_id: int) -> list[ClaimCluster]:
        rows = self._fetch_all(
            "SELECT * FROM claim_clusters WHERE run_id = ? ORDER BY cluster_score DESC, id ASC",
            (run_id,),
        )
        return [self._row_to_cluster(row) for row in rows]

    @staticmethod
    def _row_to_cluster(row: sqlite3.Row) -> ClaimCluster:
        return ClaimCluster(
            id=row["id"],
            run_id=row["run_id"],
            claim_type=row["claim_type"],
            cluster_label=row["cluster_label"],
            cluster_summary=row["cluster_summary"],
            cluster_score=row["cluster_score"],
            claim_count=row["claim_count"],
            article_count=row["article_count"],
            created_at=row["created_at"],
        )
