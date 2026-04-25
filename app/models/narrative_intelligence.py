from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ArticleAnalysisDocument:
    article_id: int
    source_id: int
    source_name: str
    source_domain: str
    title: str
    subtitle: str | None
    body_text: str
    published_at: str
    category: str | None
    chunk_texts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TopicCandidate:
    topic_id: str
    label: str
    keywords: tuple[str, ...]
    article_ids: tuple[int, ...]
    confidence: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NarrativeFrame:
    frame_id: str
    article_id: int
    topic_id: str | None
    status: str
    main_claim: str
    actors: tuple[str, ...]
    cause: str | None
    mechanism: str | None
    consequence: str | None
    future_expectation: str | None
    valence: str | None
    implications: tuple[str, ...]
    representative_quotes: tuple[str, ...] = ()
    confidence: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NarrativeFrameEmbedding:
    frame_id: str
    representation_text: str
    vector: tuple[float, ...]
    model_name: str


@dataclass(frozen=True, slots=True)
class NarrativeCluster:
    cluster_id: str
    topic_id: str | None
    frame_ids: tuple[str, ...]
    centroid_frame_id: str | None
    noise: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NarrativeClusterLabel:
    cluster_id: str
    title: str
    summary: str
    canonical_claim: str
    typical_formulations: tuple[str, ...]
    key_actors: tuple[str, ...]
    causal_chain: tuple[str, ...]
    dominant_tone: str | None
    counter_narrative: str | None
    representative_examples: tuple[str, ...]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NarrativeAssignment:
    article_id: int
    frame_id: str
    cluster_id: str | None
    similarity_score: float
    assigned: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class NarrativeDynamicsPoint:
    period_start: str
    period_end: str
    article_count: int
    share_of_corpus: float
    source_diversity: float
    mean_intensity: float | None = None
    burst_score: float | None = None


@dataclass(frozen=True, slots=True)
class NarrativeDynamicsSeries:
    cluster_id: str
    points: tuple[NarrativeDynamicsPoint, ...]
    total_articles: int
    growth_rate: float | None = None
    stability_score: float | None = None


@dataclass(frozen=True, slots=True)
class NarrativeEvaluationReport:
    topic_coherence: float | None = None
    narrative_coherence: float | None = None
    precision: float | None = None
    recall: float | None = None
    interpretability: float | None = None
    novelty_detection_quality: float | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NarrativeIntelligenceRunResult:
    documents: tuple[ArticleAnalysisDocument, ...]
    topics: tuple[TopicCandidate, ...]
    frames: tuple[NarrativeFrame, ...]
    embeddings: tuple[NarrativeFrameEmbedding, ...]
    clusters: tuple[NarrativeCluster, ...]
    labels: tuple[NarrativeClusterLabel, ...]
    assignments: tuple[NarrativeAssignment, ...]
    dynamics: tuple[NarrativeDynamicsSeries, ...]
    evaluation: NarrativeEvaluationReport | None = None
