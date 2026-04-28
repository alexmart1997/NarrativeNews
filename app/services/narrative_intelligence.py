from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import re
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
from app.repositories import (
    ArticleChunkRepository,
    ArticleRepository,
    NarrativeArticleAnalysisRepository,
    SourceRepository,
)
from app.services.llm import BaseEmbeddingClient, BaseLLMClient, LLMError, _parse_json_object


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
    min_topic_text_characters: int = 160
    min_narrative_claim_characters: int = 40
    min_cluster_size: int = 3
    min_cluster_article_support: int = 3
    min_cluster_source_support: int = 1
    classification_margin: float = 0.08
    topic_hdbscan_min_cluster_size: int = 12


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

        embeddings = _to_embedding_matrix(self.embedding_backend.embed_texts(texts))
        umap_model = (
            UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")
            if self.config.topic_reduce_dimensionality
            else None
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.config.topic_hdbscan_min_cluster_size,
            metric="euclidean",
        )
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
            keywords = _clean_topic_keywords(topic_model.get_topic(topic_id)[:12])
            if not keywords:
                keywords = _derive_topic_keywords_from_documents(documents, article_ids)
            if not keywords:
                keywords = ("общая тема",)
            topics.append(
                TopicCandidate(
                    topic_id=f"topic-{topic_id}",
                    label=_topic_label_from_keywords(keywords, topic_id),
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
        parts = [_normalize_topic_text(document.title)]
        if document.subtitle:
            parts.append(_normalize_topic_text(document.subtitle))
        chunk_texts = document.chunk_texts[:4] or (document.body_text,)
        parts.extend(_normalize_topic_text(text) for text in chunk_texts)
        merged = "\n\n".join(part for part in parts if part)
        if len(merged) < 160:
            merged = _normalize_topic_text(document.body_text)
        return merged


class LLMNarrativeFrameExtractor(NarrativeFrameExtractor):
    def __init__(self, llm_client: BaseLLMClient, config: NarrativeIntelligenceConfig | None = None) -> None:
        self.llm_client = llm_client
        self.config = config or NarrativeIntelligenceConfig()

    def extract_frames(
        self,
        document: ArticleAnalysisDocument,
        topics: Sequence[TopicCandidate] | None = None,
    ) -> list[NarrativeFrame]:
        try:
            payload = _generate_json_with_repair(
                self.llm_client,
                self._build_prompt(document, topics or ()),
                system_prompt=self._system_prompt(),
                temperature=self.config.extraction_temperature,
                max_tokens=self.config.extraction_max_tokens,
            )
        except LLMError as exc:
            return [self._fallback_frame(document, error=str(exc))]

        raw_frames = payload.get("frames", [])
        if not isinstance(raw_frames, list):
            return [self._fallback_frame(document, error="frames is not a list")]

        frames: list[NarrativeFrame] = []
        for raw_frame in raw_frames[: self.config.max_frames_per_article]:
            frame = self._parse_frame(document, raw_frame)
            if frame is not None:
                frames.append(frame)
        return frames or [self._fallback_frame(document, error="no valid frames parsed")]

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You extract narrative frames from Russian news articles. "
            "A narrative is not a topic and not a summary. "
            "A narrative is a causal-emotional interpretive story: what happens, who acts, why it happens, "
            "how it happens, what consequences follow, what future expectation is formed, and what implication "
            "or evaluative tone is present. "
            "Do not treat isolated event reports, sports results, personnel updates, ceremonial coverage, "
            "weather updates, flight restrictions, air-raid alerts, or operational bulletins as narratives unless "
            "the text explicitly builds an interpretive causal frame with implications or expectations. "
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
            "- do not return sports match results, staff reshuffles, one-off incidents, alerts, or ceremonial reports as narratives unless they are explicitly framed as evidence of a broader process\n"
            "- if the text only states what happened, who won, who was appointed, where an alert was declared, or what restrictions were introduced, return no_clear_narrative\n"
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
        main_claim = str(raw_frame.get("main_claim", "")).strip()
        actors = _coerce_string_tuple(raw_frame.get("actors"))
        implications = _coerce_string_tuple(raw_frame.get("implications"))
        quotes = _coerce_string_tuple(raw_frame.get("representative_quotes"))
        confidence = raw_frame.get("confidence")
        confidence_value = float(confidence) if isinstance(confidence, (int, float)) else None
        frame = NarrativeFrame(
            frame_id=f"frame-{uuid.uuid4().hex}",
            article_id=document.article_id,
            topic_id=str(raw_frame.get("topic_id")) if raw_frame.get("topic_id") is not None else None,
            status=status,
            main_claim=main_claim,
            actors=actors,
            cause=_coerce_optional_text(raw_frame.get("cause")),
            mechanism=_coerce_optional_text(raw_frame.get("mechanism")),
            consequence=_coerce_optional_text(raw_frame.get("consequence")),
            future_expectation=_coerce_optional_text(raw_frame.get("future_expectation")),
            valence=_coerce_optional_text(raw_frame.get("valence")),
            implications=implications,
            representative_quotes=quotes,
            confidence=confidence_value,
            metadata={"raw_json": raw_frame, "source_domain": document.source_domain},
        )
        if not _is_narrative_frame_informative(frame):
            if frame.status == "no_clear_narrative":
                return frame
            return NarrativeFrame(
                frame_id=frame.frame_id,
                article_id=frame.article_id,
                topic_id=frame.topic_id,
                status="no_clear_narrative",
                main_claim=document.title.strip() or "no clear narrative",
                actors=(),
                cause=None,
                mechanism=None,
                consequence=None,
                future_expectation=None,
                valence=None,
                implications=(),
                representative_quotes=(),
                confidence=None,
                metadata={**frame.metadata, "fallback_reason": "frame_not_informative"},
            )
        return frame

    @staticmethod
    def _fallback_frame(document: ArticleAnalysisDocument, *, error: str) -> NarrativeFrame:
        return NarrativeFrame(
            frame_id=f"frame-{uuid.uuid4().hex}",
            article_id=document.article_id,
            topic_id=None,
            status="no_clear_narrative",
            main_claim=document.title.strip() or "no clear narrative",
            actors=(),
            cause=None,
            mechanism=None,
            consequence=None,
            future_expectation=None,
            valence=None,
            implications=(),
            representative_quotes=(),
            confidence=None,
            metadata={"fallback_reason": error, "source_domain": document.source_domain},
        )


class NarrativeFrameTextFormatter:
    @staticmethod
    def to_representation_text(frame: NarrativeFrame) -> str:
        parts = [f"main_claim: {frame.main_claim}"]
        if frame.cause:
            parts.append(f"cause: {frame.cause}")
        if frame.mechanism:
            parts.append(f"mechanism: {frame.mechanism}")
        if frame.consequence:
            parts.append(f"consequence: {frame.consequence}")
        if frame.future_expectation:
            parts.append(f"future_expectation: {frame.future_expectation}")
        if frame.implications:
            parts.append(f"implications: {', '.join(frame.implications)}")
        if frame.actors:
            parts.append(f"actors: {', '.join(frame.actors)}")
        if frame.valence:
            parts.append(f"valence: {frame.valence}")
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
    def __init__(self, min_cluster_size: int = 5, min_article_support: int = 3, min_source_support: int = 1) -> None:
        self.min_cluster_size = min_cluster_size
        self.min_article_support = min_article_support
        self.min_source_support = min_source_support

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

        frame_by_id = {frame.frame_id: frame for frame in frames}
        eligible_embeddings = [
            embedding
            for embedding in embeddings
            if embedding.frame_id in frame_by_id and _is_cluster_eligible_frame(frame_by_id[embedding.frame_id])
        ]
        if not eligible_embeddings:
            return []

        clusters: list[NarrativeCluster] = []
        embeddings_by_topic: dict[str, list[NarrativeFrameEmbedding]] = defaultdict(list)
        for embedding in eligible_embeddings:
            frame = frame_by_id[embedding.frame_id]
            topic_key = frame.topic_id or f"article-{frame.article_id}"
            embeddings_by_topic[topic_key].append(embedding)

        cluster_counter = 0
        for topic_key, topic_embeddings in embeddings_by_topic.items():
            if len(topic_embeddings) < self.min_cluster_size:
                continue
            vectors = [list(embedding.vector) for embedding in topic_embeddings]
            clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, metric="euclidean")
            labels = clusterer.fit_predict(vectors)

            clusters_by_label: dict[int, list[str]] = defaultdict(list)
            for embedding, label in zip(topic_embeddings, labels, strict=False):
                clusters_by_label[int(label)].append(embedding.frame_id)

            for label, frame_ids in clusters_by_label.items():
                if label == -1:
                    continue

                article_ids = {frame_by_id[frame_id].article_id for frame_id in frame_ids if frame_id in frame_by_id}
                source_ids = {
                    frame_by_id[frame_id].metadata.get("source_domain")
                    for frame_id in frame_ids
                    if frame_id in frame_by_id
                }
                if len(article_ids) < self.min_article_support:
                    continue
                if len({source_id for source_id in source_ids if source_id}) < self.min_source_support:
                    continue

                topic_ids = Counter(
                    frame_by_id[frame_id].topic_id for frame_id in frame_ids if frame_id in frame_by_id
                )
                dominant_topic = topic_ids.most_common(1)[0][0] if topic_ids else None
                clusters.append(
                    NarrativeCluster(
                        cluster_id=f"cluster-{cluster_counter}",
                        topic_id=dominant_topic if dominant_topic not in (None, "") else (None if topic_key.startswith("article-") else topic_key),
                        frame_ids=tuple(frame_ids),
                        centroid_frame_id=frame_ids[0] if frame_ids else None,
                        noise=False,
                        metadata={
                            "article_support": len(article_ids),
                            "source_support": len({source_id for source_id in source_ids if source_id}),
                            "topic_scope": topic_key,
                        },
                    )
                )
                cluster_counter += 1
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
            try:
                payload = _generate_json_with_repair(
                    self.llm_client,
                    self._build_prompt(cluster, representative_frames),
                    system_prompt=self._system_prompt(),
                    temperature=self.config.labeling_temperature,
                    max_tokens=self.config.labeling_max_tokens,
                )
                labels.append(self._parse_label(cluster, payload))
            except LLMError as exc:
                labels.append(self._fallback_label(cluster, representative_frames, error=str(exc)))
        return labels

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You label narrative clusters for analysts. "
            "Use concise analytical language, not journalistic headlines. "
            "Return all fields in Russian. "
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
        label = NarrativeClusterLabel(
            cluster_id=cluster.cluster_id,
            title=str(payload.get("title", "")).strip(),
            summary=str(payload.get("summary", "")).strip(),
            canonical_claim=str(payload.get("canonical_claim", "")).strip(),
            typical_formulations=_coerce_string_tuple(payload.get("typical_formulations")),
            key_actors=_coerce_string_tuple(payload.get("key_actors")),
            causal_chain=_coerce_string_tuple(payload.get("causal_chain")),
            dominant_tone=_coerce_optional_text(payload.get("dominant_tone")),
            counter_narrative=_coerce_optional_text(payload.get("counter_narrative")),
            representative_examples=_coerce_string_tuple(payload.get("representative_examples")),
            metadata={"raw_json": payload},
        )
        if _label_looks_placeholder(label):
            raise LLMError("Narrative label is placeholder-like.")
        return label

    @staticmethod
    def _fallback_label(
        cluster: NarrativeCluster,
        frames: Sequence[NarrativeFrame],
        *,
        error: str,
    ) -> NarrativeClusterLabel:
        main_claims = [frame.main_claim for frame in frames if frame.main_claim.strip()]
        canonical_claim = main_claims[0] if main_claims else "No canonical claim"
        actors = []
        for frame in frames:
            actors.extend(frame.actors)
        unique_actors = tuple(dict.fromkeys(actor for actor in actors if actor))
        title = _build_cluster_title(frames, canonical_claim)
        summary = _build_cluster_summary(frames, canonical_claim)
        causal_chain = _build_cluster_causal_chain(frames)
        dominant_tone = _dominant_tone(frames)
        representative_examples = _distinct_non_placeholder(main_claims)[:3]
        return NarrativeClusterLabel(
            cluster_id=cluster.cluster_id,
            title=title or cluster.cluster_id,
            summary=summary,
            canonical_claim=canonical_claim,
            typical_formulations=representative_examples,
            key_actors=unique_actors[:6],
            causal_chain=causal_chain,
            dominant_tone=dominant_tone,
            counter_narrative=None,
            representative_examples=representative_examples,
            metadata={"fallback_reason": error},
        )


class HybridNarrativeClassifier(NarrativeClassifier):
    def __init__(
        self,
        threshold: float = 0.78,
        margin: float = 0.08,
        llm_judge: BaseLLMClient | None = None,
    ) -> None:
        self.threshold = threshold
        self.margin = margin
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
            second_best_score = -1.0
            for cluster_id, centroid_embedding in centroid_embeddings.items():
                cluster = clusters_by_id.get(cluster_id)
                if cluster is None:
                    continue
                if cluster.topic_id and frame.topic_id and cluster.topic_id != frame.topic_id:
                    continue
                score = _cosine_similarity(list(embedding.vector), list(centroid_embedding.vector))
                if score > best_score:
                    second_best_score = best_score
                    best_score = score
                    best_cluster_id = cluster_id
                elif score > second_best_score:
                    second_best_score = score

            assigned = (
                best_score >= self.threshold
                and (best_score - second_best_score) >= self.margin
                and best_cluster_id in clusters_by_id
            )
            assignments.append(
                NarrativeAssignment(
                    article_id=frame.article_id,
                    frame_id=frame.frame_id,
                    cluster_id=best_cluster_id if assigned else None,
                    similarity_score=max(best_score, 0.0),
                    assigned=assigned,
                    reason="embedding_similarity" if assigned else "unknown_or_ambiguous",
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


class CachedNarrativeIntelligencePipeline:
    def __init__(
        self,
        *,
        preprocessor: CorpusArticlePreprocessor,
        article_analysis_repository: NarrativeArticleAnalysisRepository,
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
        self.article_analysis_repository = article_analysis_repository
        self.topic_backend = topic_backend
        self.frame_extractor = frame_extractor
        self.embedding_backend = embedding_backend
        self.cluster_backend = cluster_backend
        self.labeler = labeler
        self.classifier = classifier
        self.dynamics_analyzer = dynamics_analyzer
        self.evaluator = evaluator

    def materialize_article_analyses(
        self,
        *,
        date_from: str,
        date_to: str,
        source_domains: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, int]:
        documents = self.preprocessor.load_documents(
            date_from=date_from,
            date_to=date_to,
            source_domains=source_domains,
        )
        article_ids = [document.article_id for document in documents]
        existing_ids = set() if force else self.article_analysis_repository.list_existing_article_ids(article_ids=article_ids)

        processed = 0
        reused = 0
        frames_total = 0
        for document in documents:
            if document.article_id in existing_ids and not force:
                reused += 1
                continue
            frames = self.frame_extractor.extract_frames(document, ())
            embeddings = self.embedding_backend.encode_frames(frames)
            payload_json = json.dumps(
                {
                    "frames": [_serialize_frame(frame) for frame in frames],
                    "embeddings": [_serialize_embedding(embedding) for embedding in embeddings],
                },
                ensure_ascii=False,
            )
            self.article_analysis_repository.upsert_analysis(
                article_id=document.article_id,
                source_domain=document.source_domain,
                published_at=document.published_at,
                status="completed",
                frame_count=len(frames),
                payload_json=payload_json,
            )
            processed += 1
            frames_total += len(frames)
        return {
            "documents_total": len(documents),
            "documents_processed": processed,
            "documents_reused": reused,
            "frames_created": frames_total,
        }

    def run(
        self,
        *,
        date_from: str,
        date_to: str,
        source_domains: list[str] | None = None,
        ensure_cache: bool = True,
    ) -> NarrativeIntelligenceRunResult:
        documents = self.preprocessor.load_documents(
            date_from=date_from,
            date_to=date_to,
            source_domains=source_domains,
        )
        if ensure_cache:
            self.materialize_article_analyses(
                date_from=date_from,
                date_to=date_to,
                source_domains=source_domains,
                force=False,
            )

        topics = self.topic_backend.discover_topics(documents)
        topic_by_article_id = self._topic_by_article_id(topics)
        cached_records = self.article_analysis_repository.list_by_date_range_and_sources(
            date_from=date_from,
            date_to=date_to,
            source_domains=source_domains,
        )

        frames: list[NarrativeFrame] = []
        embeddings: list[NarrativeFrameEmbedding] = []
        for record in cached_records:
            payload = json.loads(record.payload_json)
            frame_topic_id = topic_by_article_id.get(record.article_id)
            for raw_frame in payload.get("frames", []):
                frame = _deserialize_frame(raw_frame, default_topic_id=frame_topic_id)
                if frame is not None:
                    frames.append(frame)
            for raw_embedding in payload.get("embeddings", []):
                embedding = _deserialize_embedding(raw_embedding)
                if embedding is not None:
                    embeddings.append(embedding)

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

    @staticmethod
    def _topic_by_article_id(topics: Sequence[TopicCandidate]) -> dict[int, str]:
        topic_by_article_id: dict[int, str] = {}
        for topic in topics:
            for article_id in topic.article_ids:
                topic_by_article_id.setdefault(article_id, topic.topic_id)
        return topic_by_article_id


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
        cluster_backend=HDBSCANNarrativeClusterBackend(
            min_cluster_size=resolved_config.min_cluster_size,
            min_article_support=resolved_config.min_cluster_article_support,
            min_source_support=resolved_config.min_cluster_source_support,
        ),
        labeler=LLMNarrativeLabeler(llm_client=llm_client, config=resolved_config),
        classifier=HybridNarrativeClassifier(
            threshold=resolved_config.classification_threshold,
            margin=resolved_config.classification_margin,
            llm_judge=llm_client,
        ),
        dynamics_analyzer=RollingWindowNarrativeDynamicsAnalyzer(),
        evaluator=NarrativeEvaluatorStub(),
    )


def build_cached_narrative_intelligence_pipeline(
    *,
    article_repository: ArticleRepository,
    article_chunk_repository: ArticleChunkRepository,
    source_repository: SourceRepository,
    article_analysis_repository: NarrativeArticleAnalysisRepository,
    llm_client: BaseLLMClient,
    embedding_client: BaseEmbeddingClient,
    config: NarrativeIntelligenceConfig | None = None,
) -> CachedNarrativeIntelligencePipeline:
    resolved_config = config or NarrativeIntelligenceConfig()
    return CachedNarrativeIntelligencePipeline(
        preprocessor=CorpusArticlePreprocessor(
            article_repository=article_repository,
            article_chunk_repository=article_chunk_repository,
            source_repository=source_repository,
        ),
        article_analysis_repository=article_analysis_repository,
        topic_backend=BERTopicTopicDiscoveryBackend(embedding_backend=embedding_client, config=resolved_config),
        frame_extractor=LLMNarrativeFrameExtractor(llm_client=llm_client, config=resolved_config),
        embedding_backend=EmbeddingNarrativeBackend(embedding_client=embedding_client),
        cluster_backend=HDBSCANNarrativeClusterBackend(
            min_cluster_size=resolved_config.min_cluster_size,
            min_article_support=resolved_config.min_cluster_article_support,
            min_source_support=resolved_config.min_cluster_source_support,
        ),
        labeler=LLMNarrativeLabeler(llm_client=llm_client, config=resolved_config),
        classifier=HybridNarrativeClassifier(
            threshold=resolved_config.classification_threshold,
            margin=resolved_config.classification_margin,
            llm_judge=llm_client,
        ),
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


def _generate_json_with_repair(
    llm_client: BaseLLMClient,
    prompt: str,
    *,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, object]:
    raw = llm_client.generate_text(
        prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    try:
        payload = _parse_json_object(raw)
    except LLMError:
        repair_prompt = (
            "Convert the following model output into one valid JSON object only. "
            "Do not add explanations, markdown fences, comments, or extra text.\n\n"
            f"{raw}"
        )
        repaired = llm_client.generate_text(
            repair_prompt,
            system_prompt="Return one valid JSON object only.",
            temperature=0.0,
            max_tokens=max_tokens,
        )
        payload = _parse_json_object(repaired)
    if not isinstance(payload, dict):
        raise NarrativeIntelligenceError("Narrative LLM output could not be converted into a JSON object.")
    return payload


def _coerce_string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                items.append(text)
        return tuple(items)
    text = str(value).strip()
    return (text,) if text else ()


RUSSIAN_STOPWORDS = {
    "это", "как", "что", "или", "для", "при", "после", "между", "также", "который", "которая",
    "которые", "новости", "риа", "россия", "заявил", "сообщил", "сообщила", "сказал", "сказала",
    "2026", "2025",
}


def _normalize_topic_text(text: str) -> str:
    cleaned = text.lower()
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = re.sub(r"\b\d{8,}\b", " ", cleaned)
    cleaned = re.sub(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", " ", cleaned)
    cleaned = re.sub(r"\b\d+\b", " ", cleaned)
    cleaned = re.sub(r"\b[a-z]{1,4}\d+[a-z0-9-]*\b", " ", cleaned)
    cleaned = re.sub(r"\b[a-z0-9_-]+\b", " ", cleaned)
    cleaned = re.sub(r"[^\w\s\-а-яё]", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    tokens = [
        token
        for token in cleaned.split()
        if (
            len(token) >= 3
            and token not in RUSSIAN_STOPWORDS
            and not token.isdigit()
            and not re.fullmatch(r"[a-z0-9_-]+", token)
        )
    ]
    return " ".join(tokens[:220])


def _clean_topic_keywords(raw_keywords: Sequence[tuple[str, float]]) -> tuple[str, ...]:
    keywords: list[str] = []
    seen: set[str] = set()
    for raw_keyword, _score in raw_keywords:
        keyword = _normalize_topic_text(str(raw_keyword))
        if not keyword:
            continue
        keyword = keyword.split(" ", 1)[0]
        if len(keyword) < 3 or keyword in seen:
            continue
        seen.add(keyword)
        keywords.append(keyword)
        if len(keywords) >= 8:
            break
    return tuple(keywords)


def _topic_label_from_keywords(keywords: tuple[str, ...], topic_id: int) -> str:
    if not keywords:
        return f"topic {topic_id}"
    return " / ".join(keywords[:3])


def _derive_topic_keywords_from_documents(
    documents: Sequence[ArticleAnalysisDocument],
    article_ids: Sequence[int],
) -> tuple[str, ...]:
    article_id_set = set(article_ids)
    token_counts: Counter[str] = Counter()
    for document in documents:
        if document.article_id not in article_id_set:
            continue
        text = _normalize_topic_text(" ".join([document.title, document.subtitle or "", document.body_text[:400]]))
        for token in text.split():
            if len(token) >= 4:
                token_counts[token] += 1
    return tuple(token for token, _count in token_counts.most_common(8))


def _is_narrative_frame_informative(frame: NarrativeFrame) -> bool:
    if frame.status == "no_clear_narrative":
        return True
    if len(frame.main_claim.strip()) < 40:
        return False
    lower_claim = frame.main_claim.lower()
    if any(
        fragment in lower_claim
        for fragment in (
            "выиграл",
            "выиграла",
            "обыграл",
            "обыграла",
            "завоевал",
            "завоевала",
            "занял должность",
            "временно занял",
            "временно уехал",
            "воздушная тревога",
            "беспилотной опасности",
            "ограничения на прием и выпуск",
            "временные ограничения",
            "богослужение",
            "турнир",
            "матч",
            "тренер",
        )
    ):
        structure_count = sum(
            1
            for value in (frame.cause, frame.mechanism, frame.consequence, frame.future_expectation)
            if value and len(value.strip()) >= 16
        )
        if structure_count < 3:
            return False
    if any(
        fragment in lower_claim
        for fragment in (
            "произош",
            "случил",
            "взрыв",
            "заявил",
            "сообщил",
            "сообщила",
            "прошел",
            "состоялся",
        )
    ) and not any((frame.cause, frame.mechanism, frame.consequence, frame.future_expectation)):
        return False
    structure_score = sum(
        1
        for value in (frame.cause, frame.mechanism, frame.consequence, frame.future_expectation)
        if value and len(value.strip()) >= 12
    )
    actor_score = 1 if frame.actors else 0
    implication_score = 1 if frame.implications else 0
    return (structure_score + actor_score + implication_score) >= 2


def _is_cluster_eligible_frame(frame: NarrativeFrame) -> bool:
    return frame.status != "no_clear_narrative" and _is_narrative_frame_informative(frame)


def _is_placeholder_text(text: str | None) -> bool:
    if text is None:
        return True
    normalized = text.strip().lower()
    return normalized in {"", "...", "—", "n/a", "none", "unknown", "общая тема"}


def _distinct_non_placeholder(values: Sequence[str]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if _is_placeholder_text(normalized):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return tuple(output)


def _dominant_tone(frames: Sequence[NarrativeFrame]) -> str | None:
    tones = [frame.valence for frame in frames if frame.valence and not _is_placeholder_text(frame.valence)]
    if not tones:
        return None
    return Counter(tones).most_common(1)[0][0]


def _build_cluster_causal_chain(frames: Sequence[NarrativeFrame]) -> tuple[str, ...]:
    causes = _distinct_non_placeholder([frame.cause or "" for frame in frames])
    mechanisms = _distinct_non_placeholder([frame.mechanism or "" for frame in frames])
    consequences = _distinct_non_placeholder([frame.consequence or "" for frame in frames])
    chain: list[str] = []
    if causes:
        chain.append(causes[0])
    if mechanisms:
        chain.append(mechanisms[0])
    if consequences:
        chain.append(consequences[0])
    return tuple(chain)


def _build_cluster_title(frames: Sequence[NarrativeFrame], canonical_claim: str) -> str:
    implications = _distinct_non_placeholder(
        [implication for frame in frames for implication in frame.implications]
    )
    actors = _distinct_non_placeholder(
        [actor for frame in frames for actor in frame.actors]
    )
    if implications and actors:
        return f"{implications[0].capitalize()} нарратив: {actors[0]}"
    if implications:
        return f"{implications[0].capitalize()} нарратив"
    if actors:
        return actors[0][:96]
    if not _is_placeholder_text(canonical_claim):
        return canonical_claim[:96]
    return "Нарративный кластер"


def _build_cluster_summary(frames: Sequence[NarrativeFrame], canonical_claim: str) -> str:
    if not _is_placeholder_text(canonical_claim):
        return canonical_claim
    examples = _distinct_non_placeholder([frame.main_claim for frame in frames])
    if examples:
        return examples[0]
    return "Кластер объединяет близкие интерпретационные формулировки."


def _label_looks_placeholder(label: NarrativeClusterLabel) -> bool:
    if _is_placeholder_text(label.title):
        return True
    if _is_placeholder_text(label.summary):
        return True
    examples = _distinct_non_placeholder(label.representative_examples)
    return not examples


def _serialize_frame(frame: NarrativeFrame) -> dict[str, object]:
    return {
        "frame_id": frame.frame_id,
        "article_id": frame.article_id,
        "topic_id": frame.topic_id,
        "status": frame.status,
        "main_claim": frame.main_claim,
        "actors": list(frame.actors),
        "cause": frame.cause,
        "mechanism": frame.mechanism,
        "consequence": frame.consequence,
        "future_expectation": frame.future_expectation,
        "valence": frame.valence,
        "implications": list(frame.implications),
        "representative_quotes": list(frame.representative_quotes),
        "confidence": frame.confidence,
        "metadata": frame.metadata,
    }


def _serialize_embedding(embedding: NarrativeFrameEmbedding) -> dict[str, object]:
    return {
        "frame_id": embedding.frame_id,
        "representation_text": embedding.representation_text,
        "vector": list(embedding.vector),
        "model_name": embedding.model_name,
    }


def _deserialize_frame(raw_frame: object, *, default_topic_id: str | None = None) -> NarrativeFrame | None:
    if not isinstance(raw_frame, dict):
        return None
    confidence = raw_frame.get("confidence")
    confidence_value = float(confidence) if isinstance(confidence, (int, float)) else None
    topic_id = raw_frame.get("topic_id")
    return NarrativeFrame(
        frame_id=str(raw_frame.get("frame_id", "")).strip() or f"frame-{uuid.uuid4().hex}",
        article_id=int(raw_frame.get("article_id", 0)),
        topic_id=str(topic_id) if topic_id is not None else default_topic_id,
        status=str(raw_frame.get("status", "ok")),
        main_claim=str(raw_frame.get("main_claim", "")).strip(),
        actors=_coerce_string_tuple(raw_frame.get("actors")),
        cause=_coerce_optional_text(raw_frame.get("cause")),
        mechanism=_coerce_optional_text(raw_frame.get("mechanism")),
        consequence=_coerce_optional_text(raw_frame.get("consequence")),
        future_expectation=_coerce_optional_text(raw_frame.get("future_expectation")),
        valence=_coerce_optional_text(raw_frame.get("valence")),
        implications=_coerce_string_tuple(raw_frame.get("implications")),
        representative_quotes=_coerce_string_tuple(raw_frame.get("representative_quotes")),
        confidence=confidence_value,
        metadata=raw_frame.get("metadata") if isinstance(raw_frame.get("metadata"), dict) else {},
    )


def _deserialize_embedding(raw_embedding: object) -> NarrativeFrameEmbedding | None:
    if not isinstance(raw_embedding, dict):
        return None
    raw_vector = raw_embedding.get("vector")
    if not isinstance(raw_vector, list) or not raw_vector:
        return None
    try:
        vector = tuple(float(value) for value in raw_vector)
    except (TypeError, ValueError):
        return None
    return NarrativeFrameEmbedding(
        frame_id=str(raw_embedding.get("frame_id", "")).strip(),
        representation_text=str(raw_embedding.get("representation_text", "")).strip(),
        vector=vector,
        model_name=str(raw_embedding.get("model_name", "")).strip(),
    )


def _to_embedding_matrix(vectors: Sequence[Sequence[float]]) -> object:
    import numpy as np

    if not vectors:
        return np.empty((0, 0), dtype=float)
    return np.asarray(vectors, dtype=float)


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
