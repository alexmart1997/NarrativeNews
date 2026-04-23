from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SourceCreate:
    name: str
    domain: str
    base_url: str
    source_type: str
    language: str = "ru"
    is_active: bool = True


@dataclass(slots=True)
class Source(SourceCreate):
    id: int = 0
    created_at: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class ArticleCreate:
    source_id: int
    url: str
    title: str
    subtitle: str | None
    body_text: str
    published_at: str
    author: str | None = None
    category: str | None = None
    language: str = "ru"
    content_hash: str = ""
    word_count: int = 0
    is_canonical: bool = True
    duplicate_group_id: str | None = None


@dataclass(slots=True)
class Article(ArticleCreate):
    id: int = 0
    created_at: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class ClaimCreate:
    article_id: int
    claim_text: str
    normalized_claim_text: str | None
    claim_type: str
    extraction_confidence: float | None = None
    classification_confidence: float | None = None
    source_sentence: str | None = None
    source_paragraph_index: int | None = None


@dataclass(slots=True)
class Claim(ClaimCreate):
    id: int = 0
    created_at: str = ""


@dataclass(slots=True)
class NarrativeRunCreate:
    topic_text: str
    date_from: str
    date_to: str
    run_status: str
    articles_selected_count: int = 0
    claims_selected_count: int = 0
    finished_at: str | None = None


@dataclass(slots=True)
class NarrativeRun(NarrativeRunCreate):
    id: int = 0
    created_at: str = ""
