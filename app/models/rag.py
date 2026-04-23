from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ArticleChunkCreate:
    article_id: int
    chunk_index: int
    chunk_text: str
    char_start: int | None = None
    char_end: int | None = None
    token_count: int | None = None


@dataclass(slots=True)
class ArticleChunk(ArticleChunkCreate):
    id: int = 0
    created_at: str = ""


@dataclass(frozen=True, slots=True)
class ChunkSearchResult:
    chunk_id: int
    article_id: int
    chunk_index: int
    chunk_text: str
    published_at: str
    article_title: str
    match_score: int
