from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from app.db.connection import create_connection
from app.db.schema import create_schema
from app.models import ArticleCreate, SourceCreate
from app.models.narrative_intelligence import (
    ArticleAnalysisDocument,
    NarrativeCluster,
    NarrativeFrame,
    NarrativeFrameEmbedding,
)
from app.repositories import ArticleChunkRepository, ArticleRepository, SourceRepository
from app.services.chunking import ChunkingService
from app.services.narrative_intelligence import (
    CorpusArticlePreprocessor,
    EmbeddingNarrativeBackend,
    HybridNarrativeClassifier,
    LLMNarrativeLabeler,
    LLMNarrativeFrameExtractor,
    NarrativeFrameTextFormatter,
    RollingWindowNarrativeDynamicsAnalyzer,
    _coerce_string_tuple,
    _generate_json_with_repair,
    _to_embedding_matrix,
)


class StubEmbeddingClient:
    model_name = "stub-embed"

    def embed_text(self, text: str) -> list[float]:
        lowered = text.lower()
        if "inflation" in lowered:
            return [1.0, 0.0]
        return [0.0, 1.0]


class SequenceLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def generate_text(self, prompt: str, **_: object) -> str:
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


class NarrativeIntelligenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path("tests") / ".tmp" / f"narrative-intelligence-{uuid.uuid4().hex}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.temp_dir / "narrative-intelligence.db"
        self.connection = create_connection(self.database_path)
        create_schema(self.connection)
        self.source_repo = SourceRepository(self.connection)
        self.article_repo = ArticleRepository(self.connection)
        self.chunk_repo = ArticleChunkRepository(self.connection)
        self.chunking_service = ChunkingService()

        self.source = self.source_repo.create(
            SourceCreate(
                name="РИА Новости",
                domain="ria.ru",
                base_url="https://ria.ru",
                source_type="news_agency",
            )
        )

    def tearDown(self) -> None:
        self.connection.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_article(self, *, title: str, body_text: str, published_at: str):
        article = self.article_repo.create_article(
            ArticleCreate(
                source_id=self.source.id,
                url=f"https://ria.ru/{uuid.uuid4().hex}.html",
                title=title,
                subtitle=None,
                body_text=body_text,
                published_at=published_at,
                content_hash=uuid.uuid4().hex,
                word_count=50,
                is_canonical=True,
            )
        )
        self.chunk_repo.create_many(self.chunking_service.chunk_article(article))
        return article

    def test_preprocessor_uses_existing_database_articles_and_chunks(self) -> None:
        article = self._create_article(
            title="Инфляция ускорилась",
            body_text="Инфляция ускорилась в марте. Рост цен затронул продукты и услуги.",
            published_at="2026-04-10T10:00:00",
        )

        preprocessor = CorpusArticlePreprocessor(
            article_repository=self.article_repo,
            article_chunk_repository=self.chunk_repo,
            source_repository=self.source_repo,
        )

        documents = preprocessor.load_documents(
            date_from="2026-04-01T00:00:00",
            date_to="2026-04-30T23:59:59",
            source_domains=["ria.ru"],
        )

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].article_id, article.id)
        self.assertGreaterEqual(len(documents[0].chunk_texts), 1)

    def test_frame_formatter_builds_narrative_representation_not_article_text(self) -> None:
        frame = NarrativeFrame(
            frame_id="frame-1",
            article_id=1,
            topic_id="topic-1",
            status="ok",
            main_claim="Inflation is accelerating",
            actors=("Central bank", "consumers"),
            cause="tariff growth",
            mechanism="higher costs pass into retail prices",
            consequence="consumer prices increase",
            future_expectation="inflation remains elevated",
            valence="negative",
            implications=("economic",),
        )

        representation = NarrativeFrameTextFormatter.to_representation_text(frame)

        self.assertIn("main_claim: Inflation is accelerating", representation)
        self.assertIn("cause: tariff growth", representation)
        self.assertIn("future_expectation: inflation remains elevated", representation)

    def test_embedding_backend_encodes_narrative_representation(self) -> None:
        frame = NarrativeFrame(
            frame_id="frame-1",
            article_id=1,
            topic_id="topic-1",
            status="ok",
            main_claim="Inflation is accelerating",
            actors=("Central bank",),
            cause="tariff growth",
            mechanism="prices rise",
            consequence="inflation pressures",
            future_expectation="inflation remains elevated",
            valence="negative",
            implications=("economic",),
        )

        backend = EmbeddingNarrativeBackend(embedding_client=StubEmbeddingClient())
        outputs = backend.encode_frames([frame])

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].frame_id, "frame-1")
        self.assertEqual(outputs[0].model_name, "stub-embed")
        self.assertEqual(outputs[0].vector, (1.0, 0.0))

    def test_classifier_supports_unknown_frames(self) -> None:
        frame = NarrativeFrame(
            frame_id="frame-1",
            article_id=1,
            topic_id="topic-1",
            status="ok",
            main_claim="Inflation is accelerating",
            actors=("Central bank",),
            cause="tariff growth",
            mechanism="prices rise",
            consequence="inflation pressures",
            future_expectation="inflation remains elevated",
            valence="negative",
            implications=("economic",),
        )
        clusters = [
            NarrativeCluster(
                cluster_id="cluster-1",
                topic_id="topic-1",
                frame_ids=("frame-2",),
                centroid_frame_id="frame-2",
            )
        ]
        embeddings = [
            NarrativeFrameEmbedding(
                frame_id="frame-1",
                representation_text="frame 1",
                vector=(1.0, 0.0),
                model_name="stub-embed",
            ),
            NarrativeFrameEmbedding(
                frame_id="frame-2",
                representation_text="frame 2",
                vector=(0.0, 1.0),
                model_name="stub-embed",
            ),
        ]

        classifier = HybridNarrativeClassifier(threshold=0.95)
        assignments = classifier.classify_frames([frame], clusters, embeddings)

        self.assertEqual(len(assignments), 1)
        self.assertFalse(assignments[0].assigned)
        self.assertIsNone(assignments[0].cluster_id)

    def test_dynamics_analyzer_aggregates_by_period(self) -> None:
        documents = [
            ArticleAnalysisDocument(
                article_id=1,
                source_id=self.source.id,
                source_name=self.source.name,
                source_domain=self.source.domain,
                title="A",
                subtitle=None,
                body_text="Body",
                published_at="20260410T1000",
                category="Новости",
            ),
            ArticleAnalysisDocument(
                article_id=2,
                source_id=self.source.id,
                source_name=self.source.name,
                source_domain=self.source.domain,
                title="B",
                subtitle=None,
                body_text="Body",
                published_at="20260510T1000",
                category="Новости",
            ),
        ]
        frames = [
            NarrativeFrame(
                frame_id="frame-1",
                article_id=1,
                topic_id="topic-1",
                status="ok",
                main_claim="A",
                actors=("Actor",),
                cause="Cause",
                mechanism="Mechanism",
                consequence="Consequence",
                future_expectation="Future",
                valence="negative",
                implications=("economic",),
                confidence=0.8,
            ),
            NarrativeFrame(
                frame_id="frame-2",
                article_id=2,
                topic_id="topic-1",
                status="ok",
                main_claim="B",
                actors=("Actor",),
                cause="Cause",
                mechanism="Mechanism",
                consequence="Consequence",
                future_expectation="Future",
                valence="negative",
                implications=("economic",),
                confidence=0.6,
            ),
        ]
        from app.models.narrative_intelligence import NarrativeAssignment

        assignments = [
            NarrativeAssignment(article_id=1, frame_id="frame-1", cluster_id="cluster-1", similarity_score=0.9, assigned=True),
            NarrativeAssignment(article_id=2, frame_id="frame-2", cluster_id="cluster-1", similarity_score=0.85, assigned=True),
        ]

        analyzer = RollingWindowNarrativeDynamicsAnalyzer()
        series = analyzer.analyze(assignments, documents, frames)

        self.assertEqual(len(series), 1)
        self.assertEqual(series[0].cluster_id, "cluster-1")
        self.assertEqual(len(series[0].points), 2)
        self.assertEqual(series[0].total_articles, 2)

    def test_embedding_matrix_helper_returns_numpy_array(self) -> None:
        matrix = _to_embedding_matrix([[1.0, 2.0], [3.0, 4.0]])

        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(matrix.dtype.kind, "f")

    def test_string_tuple_helper_accepts_none_and_scalar_values(self) -> None:
        self.assertEqual(_coerce_string_tuple(None), ())
        self.assertEqual(_coerce_string_tuple(" actor "), ("actor",))
        self.assertEqual(_coerce_string_tuple([" a ", None, ""]), ("a",))

    def test_generate_json_with_repair_recovers_from_invalid_first_response(self) -> None:
        llm_client = SequenceLLMClient(
            [
                "not valid json",
                '{"frames": [{"status": "ok", "main_claim": "claim", "actors": null, "implications": null}]}',
            ]
        )

        payload = _generate_json_with_repair(llm_client, "prompt")

        self.assertIn("frames", payload)
        self.assertEqual(llm_client.calls, 2)

    def test_frame_extractor_uses_json_repair_flow(self) -> None:
        llm_client = SequenceLLMClient(
            [
                "garbled output",
                '{"frames": [{"status": "ok", "main_claim": "Narrative", "actors": null, "implications": null, "representative_quotes": null}]}',
            ]
        )
        extractor = LLMNarrativeFrameExtractor(llm_client=llm_client)
        document = ArticleAnalysisDocument(
            article_id=1,
            source_id=self.source.id,
            source_name=self.source.name,
            source_domain=self.source.domain,
            title="Test",
            subtitle=None,
            body_text="Body",
            published_at="20260410T1000",
            category="Новости",
        )

        frames = extractor.extract_frames(document, ())

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].main_claim, "Narrative")

    def test_frame_extractor_falls_back_to_no_clear_narrative(self) -> None:
        llm_client = SequenceLLMClient(["still not json", "still not json"])
        extractor = LLMNarrativeFrameExtractor(llm_client=llm_client)
        document = ArticleAnalysisDocument(
            article_id=1,
            source_id=self.source.id,
            source_name=self.source.name,
            source_domain=self.source.domain,
            title="Fallback title",
            subtitle=None,
            body_text="Body",
            published_at="20260410T1000",
            category="Новости",
        )

        frames = extractor.extract_frames(document, ())

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].status, "no_clear_narrative")
        self.assertEqual(frames[0].main_claim, "Fallback title")

    def test_labeler_falls_back_when_llm_cannot_return_json(self) -> None:
        llm_client = SequenceLLMClient(["bad", "bad"])
        labeler = LLMNarrativeLabeler(llm_client=llm_client)
        cluster = NarrativeCluster(
            cluster_id="cluster-1",
            topic_id="topic-1",
            frame_ids=("frame-1",),
            centroid_frame_id="frame-1",
        )
        frame = NarrativeFrame(
            frame_id="frame-1",
            article_id=1,
            topic_id="topic-1",
            status="ok",
            main_claim="Energy prices are driving inflation",
            actors=("Central bank",),
            cause="Energy shock",
            mechanism=None,
            consequence=None,
            future_expectation=None,
            valence="negative",
            implications=("economic",),
        )

        labels = labeler.label_clusters([cluster], [frame])

        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0].canonical_claim, "Energy prices are driving inflation")
        self.assertIn("fallback_reason", labels[0].metadata)


if __name__ == "__main__":
    unittest.main()
