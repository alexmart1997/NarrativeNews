from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from statistics import mean
from typing import Iterable, Sequence
import uuid

from app.models import Article
from app.models.narrative_intelligence import (
    ArticleAnalysisDocument,
    NarrativeAssignment,
    NarrativeCluster,
    NarrativeClusterLabel,
    NarrativeDynamicsPoint,
    NarrativeDynamicsSeries,
    NarrativeEvaluationReport,
    NarrativeFrame,
    NarrativeFrameEmbedding,
    NarrativeIntelligenceRunResult,
    TopicCandidate,
)
from app.repositories import ArticleChunkRepository, ArticleRepository, SourceRepository
from app.services.llm import BaseEmbeddingClient, BaseLLMClient


class NarrativeIntelligenceError(RuntimeError):
    """Raised when the narrative intelligence pipeline cannot complete a phase."""


class NarrativeIntelligenceDependencyError(NarrativeIntelligenceError):
    """Raised when an optional topic or clustering backend is not installed."""


@dataclass(frozen=True, slots=True)
class NarrativeIntelligenceConfig:
    topic_backend_name: str = "bertopic"
    topic_min_size: int = 15
    topic_reduce_dimensionality: bool = True
    topic_use_ctfidf: bool = True
    extraction_temperature: float = 0.0
    extraction_max_tokens: int = 900
    labeling_temperature: float = 0.0
    labeling_max_tokens: int = 900
    classification_threshold: float = 0.78
    dynamics_min_points: int = 2
    max_frames_per_article: int = 3


class TopicDiscoveryBackend(ABC):
    @abstractmethod
    def discover_topics(self, documents: Sequence[ArticleAnalysisDocument]) -> list[TopicCandidate]:
        raise NotImplementedError


class NarrativeFrameExtractor(ABC):
    @abstractmethod
    def extract_frames(
        self,
        document: ArticleAnalysisDocument,
        topics: Sequence[TopicCandidate] | None = None,
    ) -> list[NarrativeFrame]:
        raise NotImplementedError


class NarrativeEmbeddingBackend(ABC):
    @abstractmethod
    def encode_frames(self, frames: Sequence[NarrativeFrame]) -> list[NarrativeFrameEmbedding]:
        raise NotImplementedError


class NarrativeClusterBackend(ABC):
    @abstractmethod
    def cluster_frames(
        self,
        topics: Sequence[TopicCandidate],
        frames: Sequence[NarrativeFrame],
        embeddings: Sequence[NarrativeFrameEmbedding],
    ) -> list[NarrativeCluster]:
        raise NotImplementedError


class NarrativeLabeler(ABC):
    @abstractmethod
    def label_clusters(
        self,
        clusters: Sequence[NarrativeCluster],
        frames: Sequence[NarrativeFrame],
    ) -> list[NarrativeClusterLabel]:
        raise NotImplementedError


class NarrativeClassifier(ABC):
    @abstractmethod
    def classify_frames(
        self,
        frames: Sequence[NarrativeFrame],
        clusters: Sequence[NarrativeCluster],
        embeddings: Sequence[NarrativeFrameEmbedding],
    ) -> list[NarrativeAssignment]:
        raise NotImplementedError


class NarrativeDynamicsAnalyzer(ABC):
    @abstractmethod
    def analyze(
        self,
        assignments: Sequence[NarrativeAssignment],
        documents: Sequence[ArticleAnalysisDocument],
        frames: Sequence[NarrativeFrame],
    ) -> list[NarrativeDynamicsSeries]:
        raise NotImplementedError


class NarrativeEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        topics: Sequence[TopicCandidate],
        frames: Sequence[NarrativeFrame],
        clusters: Sequence[NarrativeCluster],
        labels: Sequence[NarrativeClusterLabel],
    ) -> NarrativeEvaluationReport:
        raise NotImplementedError


class CorpusArticlePreprocessor:
    def __init__(
        self,
        article_repository: ArticleRepository,
        article_chunk_repository: ArticleChunkRepository,
        source_repository: SourceRepository,
    ) -> None:
        self.article_repository = article_repository
        self.article_chunk_repository = article_chunk_repository
        self.source_repository = source_repository

    def load_documents(
        self,
        *,
        date_from: str,
        date_to: str,
        source_domains: list[str] | None = None,
    ) -> list[ArticleAnalysisDocument]:
        articles = self.article_repository.list_canonical_articles_by_date_range_and_sources(
            date_from=date_from,
            date_to=date_to,
            source_domains=source_domains,
        )
        sources_by_id = {source.id: source for source in self.source_repository.list()}

        documents: list[ArticleAnalysisDocument] = []
        for article in articles:
            source = sources_by_id.get(article.source_id)
            if source is None:
                continue
            chunks = self.article_chunk_repository.list_by_article_id(article.id)
            documents.append(
                ArticleAnalysisDocument(
                    article_id=article.id,
                    source_id=article.source_id,
                    source_name=source.name,
                    source_domain=source.domain,
                    title=article.title,
                    subtitle=article.subtitle,
                    body_text=article.body_text,
                    published_at=article.published_at,
                    category=article.category,
                    chunk_texts=tuple(chunk.chunk_text for chunk in chunks),
                )
            )
        return documents


class BERTopicTopicDiscoveryBackend(TopicDiscoveryBackend):
    """Conceptual default backend.

    The implementation is intentionally thin and depends on optional libraries.
    It keeps the project DB-agnostic while making the preferred methodology explicit.
    """

    def __init__(self, embedding_backend: BaseEmbeddingClient, config: NarrativeIntelligenceConfig | None = None) -> None:
        self.embedding_backend = embedding_backend
        self.config = config or NarrativeIntelligenceConfig()

    def discover_topics(self, documents: Sequence[ArticleAnalysisDocument]) -> list[TopicCandidate]:
        try:
            from bertopic import BERTopic
            from hdbscan import HDBSCAN
            from umap import UMAP
        except ImportError as exc:
            raise NarrativeIntelligenceDependencyError(
                "BERTopic/HDBSCAN/UMAP are not installed. The conceptual topic discovery layer is present, "
                "but this backend requires optional packages."
            ) from exc

        texts = [self._document_text(document) for document in documents]
        if not texts:
            return []

        embeddings = self.embedding_backend.embed_texts(texts)
        umap_model = (
            UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")
            if self.config.topic_reduce_dimensionality
            else None
        )
        hdbscan_model = HDBSCAN(min_cluster_size=self.config.topic_min_size, metric="euclidean")
        topic_model = BERTopic(
            embedding_model=None,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            calculate_probabilities=False,
            verbose=False,
        )
        topic_ids, _probabilities = topic_model.fit_transform(texts, embeddings=embeddings)

        article_ids_by_topic: dict[int, list[int]] = defaultdict(list)
        for document, topic_id in zip(documents, topic_ids, strict=False):
            article_ids_by_topic[int(topic_id)].append(document.article_id)

        topics: list[TopicCandidate] = []
        for topic_id, article_ids in article_ids_by_topic.items():
            if topic_id == -1:
                continue
            keywords = tuple(word for word, _score in topic_model.get_topic(topic_id)[:8])
            topics.append(
                TopicCandidate(
                    topic_id=f"topic-{topic_id}",
                    label=f"topic {topic_id}",
                    keywords=keywords,
                    article_ids=tuple(article_ids),
                    metadata={
                        "backend": "bertopic",
                        "ctfidf_enabled": self.config.topic_use_ctfidf,
                    },
                )
            )
        return topics

    @staticmethod
    def _document_text(document: ArticleAnalysisDocument) -> str:
        parts = [document.title]
        if document.subtitle:
            parts.append(document.subtitle)
        parts.extend(document.chunk_texts[:4] or (document.body_text,))
        return "\n\n".join(part for part in parts if part)


class LLMNarrativeFrameExtractor(NarrativeFrameExtractor):
    def __init__(self, llm_client: BaseLLMClient, config: NarrativeIntelligenceConfig | None = None) -> None:
        self.llm_client = llm_client
        self.config = config or NarrativeIntelligenceConfig()

    def extract_frames(
        self,
        document: ArticleAnalysisDocument,
        topics: Sequence[TopicCandidate] | None = None,
    ) -> list[NarrativeFrame]:
        payload = self.llm_client.generate_json(
            self._build_prompt(document, topics or ()),
            system_prompt=self._system_prompt(),
            temperature=self.config.extraction_temperature,
            max_tokens=self.config.extraction_max_tokens,
        )

        raw_frames = payload.get("frames", [])
        if not isinstance(raw_frames, list):
            raise NarrativeIntelligenceError("Narrative extractor returned invalid JSON: 'frames' must be a list.")

        frames: list[NarrativeFrame] = []
        for raw_frame in raw_frames[: self.config.max_frames_per_article]:
            frame = self._parse_frame(document, raw_frame)
            if frame is not None:
                frames.append(frame)
        return frames

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You extract narrative frames from Russian news articles. "
            "A narrative is not a topic and not a summary. "
            "A narrative is a causal-emotional interpretive story: what happens, who acts, why it happens, "
            "how it happens, what consequences follow, what future expectation is formed, and what implication "
            "or evaluative tone is present. "
            "Return valid JSON only. "
            "If a text has no clear narrative, return a single frame with status='no_clear_narrative'."
        )

    def _build_prompt(self, document: ArticleAnalysisDocument, topics: Sequence[TopicCandidate]) -> str:
        topic_lines = [
            f"- {topic.topic_id}: label={topic.label}; keywords={', '.join(topic.keywords)}"
            for topic in topics
            if document.article_id in topic.article_ids
        ]
        topic_block = "\n".join(topic_lines) if topic_lines else "- no assigned topic hints"

        return (
            "Extract one or more narrative frames from the article.\n\n"
            "Return JSON with this shape:\n"
            '{\n'
            '  "frames": [\n'
            "    {\n"
            '      "status": "ok" | "no_clear_narrative",\n'
            '      "topic_id": "topic-1" | null,\n'
            '      "main_claim": "...",\n'
            '      "actors": ["..."],\n'
            '      "cause": "...",\n'
            '      "mechanism": "...",\n'
            '      "consequence": "...",\n'
            '      "future_expectation": "...",\n'
            '      "valence": "negative|positive|fear|optimism|neutral|mixed",\n'
            '      "implications": ["economic", "political", "social"],\n'
            '      "representative_quotes": ["..."],\n'
            '      "confidence": 0.0\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- do not summarize the article\n"
            "- extract interpretation, not just facts\n"
            "- support multi-frame extraction\n"
            "- if no clear narrative exists, return one frame with status='no_clear_narrative'\n\n"
            f"Topic hints:\n{topic_block}\n\n"
            f"Title: {document.title}\n"
            f"Subtitle: {document.subtitle or ''}\n"
            f"Body:\n{document.body_text}\n"
        )

    @staticmethod
    def _parse_frame(document: ArticleAnalysisDocument, raw_frame: object) -> NarrativeFrame | None:
        if not isinstance(raw_frame, dict):
            return None
        status = str(raw_frame.get("status", "ok"))
        actors = tuple(str(item) for item in raw_frame.get("actors", []) if str(item).strip())
        implications = tuple(str(item) for item in raw_frame.get("implications", []) if str(item).strip())
        quotes = tuple(str(item) for item in raw_frame.get("representative_quotes", []) if str(item).strip())
        confidence = raw_frame.get("confidence")
        confidence_value = float(confidence) if isinstance(confidence, (int, float)) else None
        return NarrativeFrame(
            frame_id=f"frame-{uuid.uuid4().hex}",
            article_id=document.article_id,
            topic_id=str(raw_frame.get("topic_id")) if raw_frame.get("topic_id") is not None else None,
            status=status,
            main_claim=str(raw_frame.get("main_claim", "")).strip(),
            actors=actors,
            cause=_coerce_optional_text(raw_frame.get("cause")),
            mechanism=_coerce_optional_text(raw_frame.get("mechanism")),
            consequence=_coerce_optional_text(raw_frame.get("consequence")),
            future_expectation=_coerce_optional_text(raw_frame.get("future_expectation")),
            valence=_coerce_optional_text(raw_frame.get("valence")),
            implications=implications,
            representative_quotes=quotes,
            confidence=confidence_value,
            metadata={"raw_json": raw_frame},
        )


class NarrativeFrameTextFormatter:
    @staticmethod
    def to_representation_text(frame: NarrativeFrame) -> str:
        parts = [
            f"main_claim: {frame.main_claim or 'n/a'}",
            f"cause: {frame.cause or 'n/a'}",
            f"mechanism: {frame.mechanism or 'n/a'}",
            f"consequence: {frame.consequence or 'n/a'}",
            f"future_expectation: {frame.future_expectation or 'n/a'}",
            f"implications: {', '.join(frame.implications) if frame.implications else 'n/a'}",
            f"actors: {', '.join(frame.actors) if frame.actors else 'n/a'}",
            f"valence: {frame.valence or 'n/a'}",
        ]
        return "\n".join(parts)


class EmbeddingNarrativeBackend(NarrativeEmbeddingBackend):
    def __init__(self, embedding_client: BaseEmbeddingClient) -> None:
        self.embedding_client = embedding_client

    def encode_frames(self, frames: Sequence[NarrativeFrame]) -> list[NarrativeFrameEmbedding]:
        formatter = NarrativeFrameTextFormatter()
        outputs: list[NarrativeFrameEmbedding] = []
        for frame in frames:
            representation = formatter.to_representation_text(frame)
            vector = tuple(self.embedding_client.embed_text(representation))
            outputs.append(
                NarrativeFrameEmbedding(
                    frame_id=frame.frame_id,
                    representation_text=representation,
                    vector=vector,
                    model_name=self.embedding_client.model_name,
                )
            )
        return outputs


class HDBSCANNarrativeClusterBackend(NarrativeClusterBackend):
    def __init__(self, min_cluster_size: int = 5) -> None:
        self.min_cluster_size = min_cluster_size

    def cluster_frames(
        self,
        topics: Sequence[TopicCandidate],
        frames: Sequence[NarrativeFrame],
        embeddings: Sequence[NarrativeFrameEmbedding],
    ) -> list[NarrativeCluster]:
        try:
            import hdbscan
        except ImportError as exc:
            raise NarrativeIntelligenceDependencyError(
                "HDBSCAN is not installed. The conceptual narrative clustering layer is present, "
                "but this backend requires the optional package."
            ) from exc

        if not embeddings:
            return []

        vectors = [list(embedding.vector) for embedding in embeddings]
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, metric="euclidean")
        labels = clusterer.fit_predict(vectors)

        frame_by_id = {frame.frame_id: frame for frame in frames}
        clusters_by_label: dict[int, list[str]] = defaultdict(list)
        for embedding, label in zip(embeddings, labels, strict=False):
            clusters_by_label[int(label)].append(embedding.frame_id)

        clusters: list[NarrativeCluster] = []
        for label, frame_ids in clusters_by_label.items():
            if label == -1:
                for frame_id in frame_ids:
                    frame = frame_by_id.get(frame_id)
                    clusters.append(
                        NarrativeCluster(
                            cluster_id=f"noise-{frame_id}",
                            topic_id=frame.topic_id if frame else None,
                            frame_ids=(frame_id,),
                            centroid_frame_id=frame_id,
                            noise=True,
                        )
                    )
                continue

            topic_ids = Counter(
                frame_by_id[frame_id].topic_id for frame_id in frame_ids if frame_id in frame_by_id
            )
            dominant_topic = topic_ids.most_common(1)[0][0] if topic_ids else None
            clusters.append(
                NarrativeCluster(
                    cluster_id=f"cluster-{label}",
                    topic_id=dominant_topic,
                    frame_ids=tuple(frame_ids),
                    centroid_frame_id=frame_ids[0] if frame_ids else None,
                    noise=False,
                )
            )
        return clusters


class LLMNarrativeLabeler(NarrativeLabeler):
    def __init__(self, llm_client: BaseLLMClient, config: NarrativeIntelligenceConfig | None = None) -> None:
        self.llm_client = llm_client
        self.config = config or NarrativeIntelligenceConfig()

    def label_clusters(
        self,
        clusters: Sequence[NarrativeCluster],
        frames: Sequence[NarrativeFrame],
    ) -> list[NarrativeClusterLabel]:
        frames_by_id = {frame.frame_id: frame for frame in frames}
        labels: list[NarrativeClusterLabel] = []
        for cluster in clusters:
            representative_frames = [frames_by_id[frame_id] for frame_id in cluster.frame_ids[:5] if frame_id in frames_by_id]
            if not representative_frames:
                continue
            payload = self.llm_client.generate_json(
                self._build_prompt(cluster, representative_frames),
                system_prompt=self._system_prompt(),
                temperature=self.config.labeling_temperature,
                max_tokens=self.config.labeling_max_tokens,
            )
            labels.append(self._parse_label(cluster, payload))
        return labels

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You label narrative clusters for analysts. "
            "Use concise analytical language, not journalistic headlines. "
            "Return valid JSON only."
        )

    @staticmethod
    def _build_prompt(cluster: NarrativeCluster, frames: Sequence[NarrativeFrame]) -> str:
        frames_text = "\n\n".join(
            [
                f"- main_claim: {frame.main_claim}\n"
                f"  actors: {', '.join(frame.actors)}\n"
                f"  cause: {frame.cause or 'n/a'}\n"
                f"  mechanism: {frame.mechanism or 'n/a'}\n"
                f"  consequence: {frame.consequence or 'n/a'}\n"
                f"  future_expectation: {frame.future_expectation or 'n/a'}\n"
                f"  valence: {frame.valence or 'n/a'}"
                for frame in frames
            ]
        )
        return (
            "Label this cluster of narrative frames.\n\n"
            "Return JSON:\n"
            '{\n'
            '  "title": "...",\n'
            '  "summary": "...",\n'
            '  "canonical_claim": "...",\n'
            '  "typical_formulations": ["..."],\n'
            '  "key_actors": ["..."],\n'
            '  "causal_chain": ["cause", "mechanism", "consequence"],\n'
            '  "dominant_tone": "...",\n'
            '  "counter_narrative": "...",\n'
            '  "representative_examples": ["..."]\n'
            '}\n\n'
            f"Cluster id: {cluster.cluster_id}\n"
            f"Topic id: {cluster.topic_id}\n"
            f"Frames:\n{frames_text}\n"
        )

    @staticmethod
    def _parse_label(cluster: NarrativeCluster, payload: dict[str, object]) -> NarrativeClusterLabel:
        return NarrativeClusterLabel(
            cluster_id=cluster.cluster_id,
            title=str(payload.get("title", "")).strip(),
            summary=str(payload.get("summary", "")).strip(),
            canonical_claim=str(payload.get("canonical_claim", "")).strip(),
            typical_formulations=tuple(str(item) for item in payload.get("typical_formulations", []) if str(item).strip()),
            key_actors=tuple(str(item) for item in payload.get("key_actors", []) if str(item).strip()),
            causal_chain=tuple(str(item) for item in payload.get("causal_chain", []) if str(item).strip()),
            dominant_tone=_coerce_optional_text(payload.get("dominant_tone")),
            counter_narrative=_coerce_optional_text(payload.get("counter_narrative")),
            representative_examples=tuple(
                str(item) for item in payload.get("representative_examples", []) if str(item).strip()
            ),
            metadata={"raw_json": payload},
        )


class HybridNarrativeClassifier(NarrativeClassifier):
    def __init__(
        self,
        threshold: float = 0.78,
        llm_judge: BaseLLMClient | None = None,
    ) -> None:
        self.threshold = threshold
        self.llm_judge = llm_judge

    def classify_frames(
        self,
        frames: Sequence[NarrativeFrame],
        clusters: Sequence[NarrativeCluster],
        embeddings: Sequence[NarrativeFrameEmbedding],
    ) -> list[NarrativeAssignment]:
        embeddings_by_id = {embedding.frame_id: embedding for embedding in embeddings}
        clusters_by_id = {cluster.cluster_id: cluster for cluster in clusters}
        centroid_embeddings: dict[str, NarrativeFrameEmbedding] = {}
        for cluster in clusters:
            if cluster.centroid_frame_id and cluster.centroid_frame_id in embeddings_by_id:
                centroid_embeddings[cluster.cluster_id] = embeddings_by_id[cluster.centroid_frame_id]

        assignments: list[NarrativeAssignment] = []
        for frame in frames:
            embedding = embeddings_by_id.get(frame.frame_id)
            if embedding is None:
                assignments.append(
                    NarrativeAssignment(
                        article_id=frame.article_id,
                        frame_id=frame.frame_id,
                        cluster_id=None,
                        similarity_score=0.0,
                        assigned=False,
                        reason="missing_embedding",
                    )
                )
                continue

            best_cluster_id = None
            best_score = -1.0
            for cluster_id, centroid_embedding in centroid_embeddings.items():
                score = _cosine_similarity(list(embedding.vector), list(centroid_embedding.vector))
                if score > best_score:
                    best_score = score
                    best_cluster_id = cluster_id

            assigned = best_score >= self.threshold and best_cluster_id in clusters_by_id
            assignments.append(
                NarrativeAssignment(
                    article_id=frame.article_id,
                    frame_id=frame.frame_id,
                    cluster_id=best_cluster_id if assigned else None,
                    similarity_score=max(best_score, 0.0),
                    assigned=assigned,
                    reason="embedding_similarity" if assigned else "unknown_or_noise",
                )
            )
        return assignments


class RollingWindowNarrativeDynamicsAnalyzer(NarrativeDynamicsAnalyzer):
    def analyze(
        self,
        assignments: Sequence[NarrativeAssignment],
        documents: Sequence[ArticleAnalysisDocument],
        frames: Sequence[NarrativeFrame],
    ) -> list[NarrativeDynamicsSeries]:
        docs_by_article_id = {document.article_id: document for document in documents}
        frames_by_id = {frame.frame_id: frame for frame in frames}
        corpus_volume_by_period = Counter(_period_key(document.published_at) for document in documents)
        assigned_by_cluster: dict[str, list[NarrativeAssignment]] = defaultdict(list)
        for assignment in assignments:
            if assignment.assigned and assignment.cluster_id:
                assigned_by_cluster[assignment.cluster_id].append(assignment)

        results: list[NarrativeDynamicsSeries] = []
        for cluster_id, cluster_assignments in assigned_by_cluster.items():
            points: list[NarrativeDynamicsPoint] = []
            period_bucket: dict[str, list[NarrativeAssignment]] = defaultdict(list)
            for assignment in cluster_assignments:
                document = docs_by_article_id.get(assignment.article_id)
                if document is None:
                    continue
                period_bucket[_period_key(document.published_at)].append(assignment)

            for period, period_assignments in sorted(period_bucket.items()):
                article_ids = {assignment.article_id for assignment in period_assignments}
                documents_in_period = [docs_by_article_id[article_id] for article_id in article_ids if article_id in docs_by_article_id]
                source_diversity = len({document.source_domain for document in documents_in_period}) / max(
                    len(documents_in_period), 1
                )
                intensities = [
                    frames_by_id[assignment.frame_id].confidence
                    for assignment in period_assignments
                    if assignment.frame_id in frames_by_id and frames_by_id[assignment.frame_id].confidence is not None
                ]
                total_period_articles = corpus_volume_by_period.get(period, 0)
                points.append(
                    NarrativeDynamicsPoint(
                        period_start=f"{period}01",
                        period_end=f"{period}31",
                        article_count=len(article_ids),
                        share_of_corpus=(len(article_ids) / total_period_articles) if total_period_articles else 0.0,
                        source_diversity=source_diversity,
                        mean_intensity=mean(intensities) if intensities else None,
                        burst_score=_burst_score(len(article_ids), total_period_articles),
                    )
                )

            growth_rate = None
            if len(points) >= 2 and points[0].article_count > 0:
                growth_rate = (points[-1].article_count - points[0].article_count) / points[0].article_count

            stability_score = None
            if len(points) >= 2:
                counts = [point.article_count for point in points]
                volatility = sum(abs(current - previous) for previous, current in zip(counts, counts[1:], strict=False))
                stability_score = 1.0 / (1.0 + volatility)

            results.append(
                NarrativeDynamicsSeries(
                    cluster_id=cluster_id,
                    points=tuple(points),
                    total_articles=len({assignment.article_id for assignment in cluster_assignments}),
                    growth_rate=growth_rate,
                    stability_score=stability_score,
                )
            )
        return results


class NarrativeEvaluatorStub(NarrativeEvaluator):
    def evaluate(
        self,
        topics: Sequence[TopicCandidate],
        frames: Sequence[NarrativeFrame],
        clusters: Sequence[NarrativeCluster],
        labels: Sequence[NarrativeClusterLabel],
    ) -> NarrativeEvaluationReport:
        notes = (
            "Topic coherence and narrative coherence should be validated with human review.",
            "Silhouette-like embedding metrics are auxiliary only.",
            "Precision/recall require manually labeled narrative data.",
        )
        return NarrativeEvaluationReport(notes=notes)


class NarrativeIntelligencePipeline:
    def __init__(
        self,
        *,
        preprocessor: CorpusArticlePreprocessor,
        topic_backend: TopicDiscoveryBackend,
        frame_extractor: NarrativeFrameExtractor,
        embedding_backend: NarrativeEmbeddingBackend,
        cluster_backend: NarrativeClusterBackend,
        labeler: NarrativeLabeler,
        classifier: NarrativeClassifier,
        dynamics_analyzer: NarrativeDynamicsAnalyzer,
        evaluator: NarrativeEvaluator | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.topic_backend = topic_backend
        self.frame_extractor = frame_extractor
        self.embedding_backend = embedding_backend
        self.cluster_backend = cluster_backend
        self.labeler = labeler
        self.classifier = classifier
        self.dynamics_analyzer = dynamics_analyzer
        self.evaluator = evaluator

    def run(
        self,
        *,
        date_from: str,
        date_to: str,
        source_domains: list[str] | None = None,
    ) -> NarrativeIntelligenceRunResult:
        documents = self.preprocessor.load_documents(
            date_from=date_from,
            date_to=date_to,
            source_domains=source_domains,
        )
        topics = self.topic_backend.discover_topics(documents)

        frames: list[NarrativeFrame] = []
        for document in documents:
            frames.extend(self.frame_extractor.extract_frames(document, topics))

        embeddings = self.embedding_backend.encode_frames(frames)
        clusters = self.cluster_backend.cluster_frames(topics, frames, embeddings)
        labels = self.labeler.label_clusters(clusters, frames)
        assignments = self.classifier.classify_frames(frames, clusters, embeddings)
        dynamics = self.dynamics_analyzer.analyze(assignments, documents, frames)
        evaluation = self.evaluator.evaluate(topics, frames, clusters, labels) if self.evaluator else None

        return NarrativeIntelligenceRunResult(
            documents=tuple(documents),
            topics=tuple(topics),
            frames=tuple(frames),
            embeddings=tuple(embeddings),
            clusters=tuple(clusters),
            labels=tuple(labels),
            assignments=tuple(assignments),
            dynamics=tuple(dynamics),
            evaluation=evaluation,
        )


def build_default_narrative_intelligence_pipeline(
    *,
    article_repository: ArticleRepository,
    article_chunk_repository: ArticleChunkRepository,
    source_repository: SourceRepository,
    llm_client: BaseLLMClient,
    embedding_client: BaseEmbeddingClient,
    config: NarrativeIntelligenceConfig | None = None,
) -> NarrativeIntelligencePipeline:
    resolved_config = config or NarrativeIntelligenceConfig()
    return NarrativeIntelligencePipeline(
        preprocessor=CorpusArticlePreprocessor(
            article_repository=article_repository,
            article_chunk_repository=article_chunk_repository,
            source_repository=source_repository,
        ),
        topic_backend=BERTopicTopicDiscoveryBackend(embedding_backend=embedding_client, config=resolved_config),
        frame_extractor=LLMNarrativeFrameExtractor(llm_client=llm_client, config=resolved_config),
        embedding_backend=EmbeddingNarrativeBackend(embedding_client=embedding_client),
        cluster_backend=HDBSCANNarrativeClusterBackend(),
        labeler=LLMNarrativeLabeler(llm_client=llm_client, config=resolved_config),
        classifier=HybridNarrativeClassifier(threshold=resolved_config.classification_threshold, llm_judge=llm_client),
        dynamics_analyzer=RollingWindowNarrativeDynamicsAnalyzer(),
        evaluator=NarrativeEvaluatorStub(),
    )


def _period_key(published_at: str) -> str:
    compact = published_at.replace("-", "").replace(":", "").replace("T", "").replace(" ", "")
    return compact[:6] if len(compact) >= 6 else compact


def _burst_score(cluster_count: int, corpus_count: int) -> float | None:
    if corpus_count <= 0:
        return None
    return cluster_count / corpus_count


def _coerce_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right, strict=False):
        numerator += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return numerator / ((left_norm ** 0.5) * (right_norm ** 0.5))
