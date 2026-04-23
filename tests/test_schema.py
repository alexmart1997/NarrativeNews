from __future__ import annotations

import unittest
from pathlib import Path
import shutil
import uuid

from app.db.connection import create_connection
from app.db.schema import create_schema


class SchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path("tests") / ".tmp" / f"schema-{uuid.uuid4().hex}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.temp_dir / "schema.db"
        self.connection = create_connection(self.database_path)
        create_schema(self.connection)

    def tearDown(self) -> None:
        self.connection.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_schema_creates_required_tables(self) -> None:
        expected_tables = {
            "sources",
            "articles",
            "article_duplicates",
            "article_chunks",
            "claims",
            "narrative_runs",
            "claim_clusters",
            "claim_cluster_items",
            "narrative_results",
            "narrative_result_articles",
        }
        rows = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        table_names = {row["name"] for row in rows}
        self.assertTrue(expected_tables.issubset(table_names))

    def test_schema_creates_expected_indexes(self) -> None:
        rows = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
        index_names = {row["name"] for row in rows}
        self.assertIn("idx_articles_published_at", index_names)
        self.assertIn("idx_claims_article_id", index_names)
        self.assertIn("idx_claim_clusters_run_id", index_names)


if __name__ == "__main__":
    unittest.main()
