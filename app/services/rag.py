from __future__ import annotations

from dataclasses import dataclass

from app.models import Article, ChunkSearchResult, RAGAnswerResult
from app.repositories import ArticleChunkRepository, ArticleRepository
from app.services.llm import BaseLLMClient, LLMError


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
        self.llm_client = llm_client

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
        summary_text = self._generate_summary(query, chunks)
        return RAGAnswerResult(
            summary_text=summary_text,
            source_articles=source_articles[:5],
            top_chunks=chunks if include_debug_chunks else None,
        )

    def _generate_summary(self, query: str, chunks: list[ChunkSearchResult]) -> str:
        if not chunks:
            return "По выбранному периоду релевантные фрагменты не найдены."
        if self.llm_client is None:
            return self._fallback_summary(chunks)

        prompt = self._build_prompt(query, chunks)
        system_prompt = (
            "Ты помогаешь с news RAG. Отвечай кратко, по-русски, только на основе переданных фрагментов. "
            "Не придумывай факты и не пересказывай каждую статью отдельно."
        )
        try:
            text = self.llm_client.generate_text(
                prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=220,
            )
        except LLMError:
            return self._fallback_summary(chunks)

        return text.strip() or self._fallback_summary(chunks)

    @staticmethod
    def _build_prompt(query: str, chunks: list[ChunkSearchResult]) -> str:
        lines = [f"Запрос: {query}", "", "Фрагменты контекста:"]
        for index, chunk in enumerate(chunks[:5], start=1):
            lines.append(f"{index}. [{chunk.article_title}] {chunk.chunk_text}")
        lines.append("")
        lines.append("Сделай краткую выжимку в 1-2 абзаца только по этому контексту.")
        return "\n".join(lines)

    @staticmethod
    def _fallback_summary(chunks: list[ChunkSearchResult]) -> str:
        paragraphs: list[str] = []
        for chunk in chunks[:2]:
            text = chunk.chunk_text.strip()
            if text:
                paragraphs.append(text)
        return "\n\n".join(paragraphs) if paragraphs else "По выбранному периоду релевантные фрагменты не найдены."

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
