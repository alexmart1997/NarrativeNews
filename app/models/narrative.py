from __future__ import annotations

from dataclasses import dataclass

from app.models.entities import Article, Claim


@dataclass(slots=True)
class ClaimClusterCreate:
    run_id: int
    claim_type: str
    cluster_label: str
    cluster_summary: str | None
    cluster_score: float
    claim_count: int
    article_count: int


@dataclass(slots=True)
class ClaimCluster(ClaimClusterCreate):
    id: int = 0
    created_at: str = ""


@dataclass(slots=True)
class ClaimClusterItemCreate:
    cluster_id: int
    claim_id: int
    membership_score: float | None = None
    is_representative: bool = False


@dataclass(slots=True)
class NarrativeResultCreate:
    run_id: int
    narrative_type: str
    title: str
    formulation: str
    explanation: str | None
    strength_score: float | None = None


@dataclass(slots=True)
class NarrativeResult(NarrativeResultCreate):
    id: int = 0
    created_at: str = ""


@dataclass(slots=True)
class NarrativeResultArticleCreate:
    narrative_result_id: int
    article_id: int
    rank: int
    selection_reason: str | None = None


@dataclass(frozen=True, slots=True)
class GroupedClaimCluster:
    claim_type: str
    representative_text: str
    claims: list[Claim]
    articles: list[Article]
    cluster_score: float
