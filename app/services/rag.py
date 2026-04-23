from __future__ import annotations

from dataclasses import dataclass

from app.models import Article, ChunkSearchResult, RAGAnswerResult
from app.repositories import ArticleChunkRepository, ArticleRepository
from app.services.llm import BaseLLMClient, SimpleExtractiveLLMClient


@dataclass(frozen=True, slots=True)
class RAGSearchResult:
    chunks: list[ChunkSearchResult]
    articles: list[Article]


class RAGService:
    def __init__(
        self,
        article_chunk_repository: ArticleChunkRepository,
        article_repository: ArticleRepository,
        llm_client: BaseLLMClient | None = None,
    ) -> None:
        self.article_chunk_repository = article_chunk_repository
        self.article_repository = article_repository
        self.llm_client = llm_client or SimpleExtractiveLLMClient()

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
        articles = self._select_source_articles(chunks, max_articles=5)
        return RAGSearchResult(chunks=chunks, articles=articles)

    def answer(
        self,
        query: str,
        date_from: str,
        date_to: str,
        limit: int = 10,
        include_debug_chunks: bool = False,
    ) -> RAGAnswerResult:
        chunks = self.search_chunks(query, date_from, date_to, limit)
        source_articles = self._select_source_articles(chunks, max_articles=5)
        summary_text = self.llm_client.generate_answer(query, chunks)
        return RAGAnswerResult(
            summary_text=summary_text,
            source_articles=source_articles[:5],
            top_chunks=chunks if include_debug_chunks else None,
        )

    def _select_source_articles(
        self,
        chunks: list[ChunkSearchResult],
        *,
        max_articles: int,
    ) -> list[Article]:
        article_ids: list[int] = []
        seen: set[int] = set()
        for chunk in chunks:
            if chunk.article_id in seen:
                continue
            seen.add(chunk.article_id)
            article_ids.append(chunk.article_id)
            if len(article_ids) >= max_articles:
                break

        articles: list[Article] = []
        for article_id in article_ids:
            article = self.article_repository.get_article_by_id(article_id)
            if article is not None:
                articles.append(article)
        return articles
