from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NarrativeAnalysisRun:
    id: int
    source_domains_key: str
    date_from: str
    date_to: str
    status: str
    documents_count: int
    topics_count: int
    frames_count: int
    clusters_count: int
    labels_count: int
    assignments_count: int
    dynamics_count: int
    payload_json: str
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class NarrativeArticleAnalysis:
    id: int
    article_id: int
    source_domain: str
    published_at: str
    status: str
    frame_count: int
    payload_json: str
    created_at: str
    updated_at: str
