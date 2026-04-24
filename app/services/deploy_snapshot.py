from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3

from app.db.connection import create_connection
from app.db.schema import create_schema


@dataclass(frozen=True, slots=True)
class DeploySnapshotResult:
    output_path: Path
    articles_count: int
    chunks_count: int
    claims_count: int
    embeddings_count: int
    sources_count: int


class DeploySnapshotService:
    def create_snapshot(
        self,
        *,
        source_database_path: Path,
        output_database_path: Path,
        date_from: str,
        date_to: str,
        keep_embeddings: bool = True,
        clear_narratives: bool = False,
    ) -> DeploySnapshotResult:
        source_database_path = source_database_path.resolve()
        output_database_path = output_database_path.resolve()
        if source_database_path == output_database_path:
            raise ValueError("Output database path must be different from source database path.")

        output_database_path.parent.mkdir(parents=True, exist_ok=True)
        if output_database_path.exists():
            output_database_path.unlink()

        with create_connection(source_database_path) as source_connection, create_connection(output_database_path) as snapshot_connection:
            create_schema(snapshot_connection)
            snapshot_connection.execute("ATTACH DATABASE ? AS source_db", (str(source_database_path),))
            self._copy_sources(source_connection, snapshot_connection, date_from=date_from, date_to=date_to)
            self._copy_articles(source_connection, snapshot_connection, date_from=date_from, date_to=date_to)
            self._copy_article_duplicates(snapshot_connection)
            self._copy_article_chunks(snapshot_connection)
            if keep_embeddings:
                self._copy_chunk_embeddings(snapshot_connection)
            self._copy_claims(snapshot_connection)
            if not clear_narratives:
                self._copy_narrative_tables(snapshot_connection, date_from=date_from, date_to=date_to)
            snapshot_connection.execute("DETACH DATABASE source_db")
            snapshot_connection.execute("VACUUM")
            snapshot_connection.commit()
            return DeploySnapshotResult(
                output_path=output_database_path,
                articles_count=self._count(snapshot_connection, "articles"),
                chunks_count=self._count(snapshot_connection, "article_chunks"),
                claims_count=self._count(snapshot_connection, "claims"),
                embeddings_count=self._count(snapshot_connection, "article_chunk_embeddings"),
                sources_count=self._count(snapshot_connection, "sources"),
            )

    @staticmethod
    def _copy_sources(
        source_connection: sqlite3.Connection,
        snapshot_connection: sqlite3.Connection,
        *,
        date_from: str,
        date_to: str,
    ) -> None:
        rows = source_connection.execute(
            """
            SELECT DISTINCT s.*
            FROM sources s
            INNER JOIN articles a ON a.source_id = s.id
            WHERE a.published_at BETWEEN ? AND ?
            ORDER BY s.id ASC
            """,
            (date_from, date_to),
        ).fetchall()
        for row in rows:
            snapshot_connection.execute(
                """
                INSERT INTO sources (
                    id, name, domain, base_url, source_type, language, is_active, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    row["name"],
                    row["domain"],
                    row["base_url"],
                    row["source_type"],
                    row["language"],
                    row["is_active"],
                    row["created_at"],
                    row["updated_at"],
                ),
            )
        snapshot_connection.commit()

    @staticmethod
    def _copy_articles(
        source_connection: sqlite3.Connection,
        snapshot_connection: sqlite3.Connection,
        *,
        date_from: str,
        date_to: str,
    ) -> None:
        rows = source_connection.execute(
            """
            SELECT *
            FROM articles
            WHERE published_at BETWEEN ? AND ?
            ORDER BY id ASC
            """,
            (date_from, date_to),
        ).fetchall()
        for row in rows:
            snapshot_connection.execute(
                """
                INSERT INTO articles (
                    id, source_id, url, title, subtitle, body_text, published_at, author, category,
                    language, content_hash, word_count, is_canonical, duplicate_group_id, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    row["source_id"],
                    row["url"],
                    row["title"],
                    row["subtitle"],
                    row["body_text"],
                    row["published_at"],
                    row["author"],
                    row["category"],
                    row["language"],
                    row["content_hash"],
                    row["word_count"],
                    row["is_canonical"],
                    row["duplicate_group_id"],
                    row["created_at"],
                    row["updated_at"],
                ),
            )
        snapshot_connection.commit()

    @staticmethod
    def _copy_article_duplicates(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            INSERT INTO article_duplicates (
                id, duplicate_group_id, article_id, duplicate_type, is_primary, similarity_score, created_at
            )
            SELECT d.id, d.duplicate_group_id, d.article_id, d.duplicate_type, d.is_primary, d.similarity_score, d.created_at
            FROM source_db.article_duplicates d
            INNER JOIN articles a ON a.id = d.article_id
            """
        )
        connection.commit()

    @staticmethod
    def _copy_article_chunks(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            INSERT INTO article_chunks (
                id, article_id, chunk_index, chunk_text, char_start, char_end, token_count, created_at
            )
            SELECT c.id, c.article_id, c.chunk_index, c.chunk_text, c.char_start, c.char_end, c.token_count, c.created_at
            FROM source_db.article_chunks c
            INNER JOIN articles a ON a.id = c.article_id
            ORDER BY c.id ASC
            """
        )
        connection.commit()

    @staticmethod
    def _copy_chunk_embeddings(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            INSERT INTO article_chunk_embeddings (
                id, chunk_id, model_name, embedding_json, dimension, created_at
            )
            SELECT e.id, e.chunk_id, e.model_name, e.embedding_json, e.dimension, e.created_at
            FROM source_db.article_chunk_embeddings e
            INNER JOIN article_chunks c ON c.id = e.chunk_id
            ORDER BY e.id ASC
            """
        )
        connection.commit()

    @staticmethod
    def _copy_claims(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            INSERT INTO claims (
                id, article_id, claim_text, normalized_claim_text, claim_type,
                extraction_confidence, classification_confidence, source_sentence, source_paragraph_index, created_at
            )
            SELECT c.id, c.article_id, c.claim_text, c.normalized_claim_text, c.claim_type,
                   c.extraction_confidence, c.classification_confidence, c.source_sentence, c.source_paragraph_index, c.created_at
            FROM source_db.claims c
            INNER JOIN articles a ON a.id = c.article_id
            ORDER BY c.id ASC
            """
        )
        connection.commit()

    @staticmethod
    def _copy_narrative_tables(connection: sqlite3.Connection, *, date_from: str, date_to: str) -> None:
        connection.execute(
            """
            INSERT INTO narrative_runs (
                id, topic_text, date_from, date_to, run_status, articles_selected_count,
                claims_selected_count, created_at, finished_at
            )
            SELECT id, topic_text, date_from, date_to, run_status, articles_selected_count,
                   claims_selected_count, created_at, finished_at
            FROM source_db.narrative_runs
            WHERE date_from >= ? AND date_to <= ?
            """,
            (date_from, date_to),
        )
        connection.execute(
            """
            INSERT INTO claim_clusters (
                id, run_id, claim_type, cluster_label, cluster_summary, cluster_score,
                claim_count, article_count, created_at
            )
            SELECT cc.id, cc.run_id, cc.claim_type, cc.cluster_label, cc.cluster_summary, cc.cluster_score,
                   cc.claim_count, cc.article_count, cc.created_at
            FROM source_db.claim_clusters cc
            INNER JOIN narrative_runs nr ON nr.id = cc.run_id
            """
        )
        connection.execute(
            """
            INSERT INTO claim_cluster_items (
                id, cluster_id, claim_id, membership_score, is_representative, created_at
            )
            SELECT cci.id, cci.cluster_id, cci.claim_id, cci.membership_score, cci.is_representative, cci.created_at
            FROM source_db.claim_cluster_items cci
            INNER JOIN claim_clusters cc ON cc.id = cci.cluster_id
            INNER JOIN claims c ON c.id = cci.claim_id
            """
        )
        connection.execute(
            """
            INSERT INTO narrative_results (
                id, run_id, narrative_type, title, formulation, explanation, strength_score, created_at
            )
            SELECT nr.id, nr.run_id, nr.narrative_type, nr.title, nr.formulation, nr.explanation, nr.strength_score, nr.created_at
            FROM source_db.narrative_results nr
            INNER JOIN narrative_runs runs ON runs.id = nr.run_id
            """
        )
        connection.execute(
            """
            INSERT INTO narrative_result_articles (
                id, narrative_result_id, article_id, rank, selection_reason
            )
            SELECT nra.id, nra.narrative_result_id, nra.article_id, nra.rank, nra.selection_reason
            FROM source_db.narrative_result_articles nra
            INNER JOIN narrative_results nr ON nr.id = nra.narrative_result_id
            INNER JOIN articles a ON a.id = nra.article_id
            """
        )
        connection.commit()

    @staticmethod
    def _count(connection: sqlite3.Connection, table_name: str) -> int:
        row = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        return int(row[0]) if row is not None else 0
