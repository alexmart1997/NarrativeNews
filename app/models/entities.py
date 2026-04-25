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
