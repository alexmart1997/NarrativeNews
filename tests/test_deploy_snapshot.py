from __future__ import annotations

from pathlib import Path
import shutil
import unittest
import uuid

from app.db.connection import create_connection
from app.db.schema import create_schema
from app.models import (
    ArticleCreate,
    ClaimCreate,
    NarrativeResultArticleCreate,
    NarrativeResultCreate,
    NarrativeRunCreate,
    SourceCreate,
)
from app.repositories import (
    ArticleChunkRepository,
    ArticleRepository,
    ClaimRepository,
    NarrativeResultRepository,
    NarrativeRunRepository,
    SourceRepository,
)
from app.services import ChunkingService, DeploySnapshotService


class DeploySnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path("tests") / ".tmp" / f"deploy-snapshot-{uuid.uuid4().hex}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.temp_dir / "source.db"
        self.output_path = self.temp_dir / "deploy.db"
        self.connection = create_connection(self.database_path)
        create_schema(self.connection)
        self.source_repo = SourceRepository(self.connection)
        self.article_repo = ArticleRepository(self.connection)
        self.chunk_repo = ArticleChunkRepository(self.connection)
        self.claim_repo = ClaimRepository(self.connection)
        self.run_repo = NarrativeRunRepository(self.connection)
        self.result_repo = NarrativeResultRepository(self.connection)
        self.chunking_service = ChunkingService()

    def tearDown(self) -> None:
        self.connection.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_snapshot_keeps_only_selected_date_range(self) -> None:
        kept_article_id = self._seed_article(
            domain=f"kept-{uuid.uuid4().hex[:8]}.ru",
            url_suffix="kept",
            published_at="2025-07-10T10:00:00",
        )
        self._seed_article(
            domain=f"dropped-{uuid.uuid4().hex[:8]}.ru",
            url_suffix="dropped",
            published_at="2025-10-10T10:00:00",
        )
        self._seed_narrative_data(kept_article_id)

        service = DeploySnapshotService()
        result = service.create_snapshot(
            source_database_path=self.database_path,
            output_database_path=self.output_path,
            date_from="2025-06-01T00:00:00",
            date_to="2025-08-31T23:59:59",
        )

        self.assertEqual(result.articles_count, 1)
        self.assertGreaterEqual(result.chunks_count, 1)
        self.assertGreaterEqual(result.claims_count, 1)
        self.assertGreaterEqual(result.embeddings_count, 1)
        self.assertEqual(result.sources_count, 1)

        with create_connection(self.output_path) as snapshot_connection:
            article_ids = [row["id"] for row in snapshot_connection.execute("SELECT id FROM articles").fetchall()]
            self.assertEqual(article_ids, [kept_article_id])
            self.assertEqual(
                snapshot_connection.execute("SELECT COUNT(*) FROM narrative_runs").fetchone()[0],
                1,
            )
            self.assertEqual(
                snapshot_connection.execute("SELECT COUNT(*) FROM narrative_results").fetchone()[0],
                1,
            )

    def test_create_snapshot_can_drop_embeddings_and_clear_narratives(self) -> None:
        kept_article_id = self._seed_article(
            domain=f"kept-{uuid.uuid4().hex[:8]}.ru",
            url_suffix="kept",
            published_at="2025-07-10T10:00:00",
        )
        self._seed_narrative_data(kept_article_id)

        service = DeploySnapshotService()
        result = service.create_snapshot(
            source_database_path=self.database_path,
            output_database_path=self.output_path,
            date_from="2025-06-01T00:00:00",
            date_to="2025-08-31T23:59:59",
            keep_embeddings=False,
            clear_narratives=True,
        )

        self.assertEqual(result.embeddings_count, 0)
        with create_connection(self.output_path) as snapshot_connection:
            self.assertEqual(
                snapshot_connection.execute("SELECT COUNT(*) FROM narrative_runs").fetchone()[0],
                0,
            )
            self.assertEqual(
                snapshot_connection.execute("SELECT COUNT(*) FROM narrative_results").fetchone()[0],
                0,
            )

    def _seed_article(self, *, domain: str, url_suffix: str, published_at: str) -> int:
        source = self.source_repo.create(
            SourceCreate(
                name=domain,
                domain=domain,
                base_url=f"https://{domain}",
                source_type="news_site",
            )
        )
        article = self.article_repo.create_article(
            ArticleCreate(
                source_id=source.id,
                url=f"https://{domain}/articles/{url_suffix}",
                title=f"Article {url_suffix}",
                subtitle=None,
                body_text=(
                    "Министр заявил, что поставки вырастут к осени. "
                    "Это должно снизить дефицит на рынке.\n\n"
                    "Аналитики считают, что меры приведут к снижению цен."
                ),
                published_at=published_at,
                content_hash=uuid.uuid4().hex,
                word_count=20,
                is_canonical=True,
            )
        )
        created_chunks = self.chunk_repo.create_many(self.chunking_service.chunk_article(article))
        if created_chunks:
            self.connection.execute(
                """
                INSERT INTO article_chunk_embeddings (chunk_id, model_name, embedding_json, dimension)
                VALUES (?, ?, ?, ?)
                """,
                (created_chunks[0].id, "nomic-embed-text", "[0.1, 0.2]", 2),
            )
        self.claim_repo.create_many(
            [
                ClaimCreate(
                    article_id=article.id,
                    claim_text="Министр заявил, что поставки вырастут к осени.",
                    normalized_claim_text="Поставки вырастут к осени",
                    claim_type="predictive",
                    extraction_confidence=0.9,
                    classification_confidence=0.88,
                    source_sentence="Министр заявил, что поставки вырастут к осени.",
                    source_paragraph_index=0,
                )
            ]
        )
        self.connection.commit()
        return article.id

    def _seed_narrative_data(self, article_id: int) -> None:
        run = self.run_repo.create(
            NarrativeRunCreate(
                topic_text="рынок",
                date_from="2025-07-01T00:00:00",
                date_to="2025-07-31T23:59:59",
                run_status="completed",
                articles_selected_count=1,
                claims_selected_count=1,
            )
        )
        narrative_result = self.result_repo.create(
            NarrativeResultCreate(
                run_id=run.id,
                narrative_type="predictive",
                title="Рост поставок",
                formulation="Поставки вырастут к осени.",
                explanation="Поддержано claim и статьёй.",
                strength_score=0.7,
            )
        )
        self.result_repo.create_result_articles(
            [
                NarrativeResultArticleCreate(
                    narrative_result_id=narrative_result.id,
                    article_id=article_id,
                    rank=1,
                    selection_reason="representative",
                )
            ]
        )


if __name__ == "__main__":
    unittest.main()
