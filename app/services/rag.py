from __future__ import annotations

from dataclasses import dataclass

from app.models import Article, ChunkSearchResult
from app.repositories import ArticleChunkRepository, ArticleRepository


@dataclass(frozen=True, slots=True)
class RAGSearchResult:
    chunks: list[ChunkSearchResult]
    articles: list[Article]


class RAGService:
    def __init__(
        self,
        article_chunk_repository: ArticleChunkRepository,
        article_repository: ArticleRepository,
    ) -> None:
        self.article_chunk_repository = article_chunk_repository
        self.article_repository = article_repository

    def search_chunks(
        self,
        query: str,
        date_from: str,
        date_to: str,
        limit: int = 10,
    ) -> list[ChunkSearchResult]:
        return self.article_chunk_repository.search_chunks(query, date_from, date_to, limit)

    def search(
        self,
        query: str,
        date_from: str,
        date_to: str,
        limit: int = 10,
    ) -> RAGSearchResult:
        chunks = self.search_chunks(query, date_from, date_to, limit)
        article_ids: list[int] = []
        seen: set[int] = set()
        for chunk in chunks:
            if chunk.article_id in seen:
                continue
            seen.add(chunk.article_id)
            article_ids.append(chunk.article_id)

        articles: list[Article] = []
        for article_id in article_ids:
            article = self.article_repository.get_article_by_id(article_id)
            if article is not None:
                articles.append(article)

        return RAGSearchResult(chunks=chunks, articles=articles)
