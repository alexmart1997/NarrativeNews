from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re

from app.models import Article, ChunkSearchResult, EmbeddedChunkCandidate, RAGAnswerResult
from app.repositories import ArticleChunkRepository, ArticleRepository
from app.services.llm import BaseEmbeddingClient, BaseLLMClient


TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{2,}")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


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
        embedding_client: BaseEmbeddingClient | None = None,
        hybrid_limit: int = 24,
        rerank_limit: int = 8,
    ) -> None:
        self.article_chunk_repository = article_chunk_repository
        self.article_repository = article_repository
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.hybrid_limit = hybrid_limit
        self.rerank_limit = rerank_limit

    def search_chunks(
        self,
        query: str,
        date_from: str,
        date_to: str,
        limit: int = 10,
    ) -> list[ChunkSearchResult]:
        candidates = self._hybrid_retrieve(query, date_from, date_to, limit=max(limit, self.hybrid_limit))
        reranked = self._rerank(query, candidates, limit=min(limit, self.rerank_limit))
        return reranked[:limit]

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

    def _hybrid_retrieve(
        self,
        query: str,
        date_from: str,
        date_to: str,
        *,
        limit: int,
    ) -> list[ChunkSearchResult]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        lexical_rows = self.article_chunk_repository.search_chunks_lexical(
            query,
            date_from,
            date_to,
            limit=limit,
        )
        merged: dict[int, ChunkSearchResult] = {}
        lexical_hit = False
        for rank, row in enumerate(lexical_rows, start=1):
            lexical_score = 1.0 / rank
            overlap_score = self._token_overlap_score(query_tokens, row.article_title, row.chunk_text)
            lexical_hit = lexical_hit or overlap_score > 0
            merged[row.chunk_id] = replace(
                row,
                lexical_score=lexical_score,
                final_score=0.65 * lexical_score + 0.10 * overlap_score,
            )

        vector_rows: list[ChunkSearchResult] = []
        if self.embedding_client is not None:
            try:
                query_embedding = self.embedding_client.embed_text(query)
            except Exception:
                query_embedding = []
            if query_embedding:
                vector_rows = self._search_vector_candidates(
                    query_embedding=query_embedding,
                    query_tokens=query_tokens,
                    date_from=date_from,
                    date_to=date_to,
                    limit=limit,
                )

        for rank, row in enumerate(vector_rows, start=1):
            vector_rank_bonus = 1.0 / rank
            overlap_score = self._token_overlap_score(query_tokens, row.article_title, row.chunk_text)
            existing = merged.get(row.chunk_id)
            if existing is None:
                final_score = 0.45 * row.vector_score + 0.15 * vector_rank_bonus + 0.15 * overlap_score
                merged[row.chunk_id] = replace(
                    row,
                    final_score=final_score,
                )
                continue

            combined_vector = max(existing.vector_score, row.vector_score)
            combined_final = (
                existing.final_score
                + 0.25 * row.vector_score
                + 0.05 * vector_rank_bonus
                + 0.05 * overlap_score
            )
            merged[row.chunk_id] = replace(
                existing,
                vector_score=combined_vector,
                final_score=combined_final,
            )

        ranked = sorted(
            merged.values(),
            key=lambda item: (item.final_score, item.lexical_score, item.vector_score, item.published_at),
            reverse=True,
        )

        if not ranked:
            return []

        if lexical_hit:
            ranked = [item for item in ranked if item.lexical_score > 0 or item.final_score >= 0.2]
        else:
            ranked = [item for item in ranked if item.vector_score >= 0.55 or item.final_score >= 0.35]

        top = ranked[:limit]
        if not self._has_topical_match(query_tokens, top):
            return []
        return top

    def _search_vector_candidates(
        self,
        *,
        query_embedding: list[float],
        query_tokens: list[str],
        date_from: str,
        date_to: str,
        limit: int,
    ) -> list[ChunkSearchResult]:
        if self.embedding_client is None:
            return []

        candidates = self.article_chunk_repository.list_vector_candidates(
            model_name=self.embedding_client.model_name,
            date_from=date_from,
            date_to=date_to,
        )
        scored: list[ChunkSearchResult] = []
        for candidate in candidates:
            similarity = self._cosine_similarity(query_embedding, candidate.embedding)
            if similarity <= 0:
                continue
            overlap_score = self._token_overlap_score(query_tokens, candidate.article_title, candidate.chunk_text)
            final_score = 0.55 * similarity + 0.15 * overlap_score
            scored.append(
                ChunkSearchResult(
                    chunk_id=candidate.chunk_id,
                    article_id=candidate.article_id,
                    chunk_index=candidate.chunk_index,
                    chunk_text=candidate.chunk_text,
                    published_at=candidate.published_at,
                    article_title=candidate.article_title,
                    match_score=similarity,
                    vector_score=similarity,
                    final_score=final_score,
                )
            )
        scored.sort(key=lambda item: (item.vector_score, item.final_score, item.published_at), reverse=True)
        return scored[:limit]

    def _rerank(
        self,
        query: str,
        candidates: list[ChunkSearchResult],
        *,
        limit: int,
    ) -> list[ChunkSearchResult]:
        if not candidates:
            return []

        initial = candidates[: max(limit, self.rerank_limit)]
        if self.llm_client is None:
            return initial[:limit]

        prompt_lines = [
            f"Запрос: {query}",
            "",
            "Выбери самые релевантные фрагменты. Верни JSON вида {\"ranked_chunk_ids\": [id1, id2]}",
            "В список включай только действительно релевантные запросу фрагменты.",
            "",
            "Кандидаты:",
        ]
        for item in initial:
            prompt_lines.append(f"- id={item.chunk_id} | title={item.article_title} | text={item.chunk_text}")
        prompt = "\n".join(prompt_lines)
        system_prompt = (
            "Ты делаешь reranking новостных фрагментов для RAG. "
            "Отвечай только JSON-объектом. Не добавляй пояснений. "
            "Не включай нерелевантные фрагменты."
        )
        try:
            payload = self.llm_client.generate_json(
                prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=180,
            )
        except Exception:
            return initial[:limit]

        ranked_ids = payload.get("ranked_chunk_ids", [])
        if not isinstance(ranked_ids, list):
            return initial[:limit]

        by_id = {item.chunk_id: item for item in initial}
        reranked: list[ChunkSearchResult] = []
        used: set[int] = set()
        for index, raw_chunk_id in enumerate(ranked_ids, start=1):
            try:
                chunk_id = int(raw_chunk_id)
            except (TypeError, ValueError):
                continue
            item = by_id.get(chunk_id)
            if item is None or chunk_id in used:
                continue
            used.add(chunk_id)
            reranked.append(
                replace(
                    item,
                    rerank_score=1.0 / index,
                    final_score=item.final_score + (0.35 / index),
                )
            )

        if not reranked:
            return initial[:limit]

        for item in initial:
            if item.chunk_id in used:
                continue
            reranked.append(item)
        reranked.sort(
            key=lambda item: (item.rerank_score, item.final_score, item.lexical_score, item.vector_score),
            reverse=True,
        )
        return reranked[:limit]

    def _generate_summary(self, query: str, chunks: list[ChunkSearchResult]) -> str:
        if not chunks:
            return "По выбранному запросу релевантные фрагменты не найдены."
        if self.llm_client is None:
            return self._fallback_summary(chunks)

        prompt = self._build_prompt(query, chunks)
        system_prompt = (
            "Ты помогаешь с новостным RAG. Отвечай только на русском языке. "
            "Используй только факты из переданных фрагментов. "
            "Не придумывай детали, не добавляй факты вне контекста и не смешивай нерелевантные сюжеты. "
            "Если данных мало, скажи об этом кратко."
        )
        try:
            text = self.llm_client.generate_text(
                prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=260,
            )
        except Exception:
            return self._fallback_summary(chunks)

        cleaned = text.strip()
        if not cleaned or CHINESE_RE.search(cleaned):
            return self._fallback_summary(chunks)
        return cleaned

    @staticmethod
    def _build_prompt(query: str, chunks: list[ChunkSearchResult]) -> str:
        lines = [
            f"Запрос: {query}",
            "",
            "Релевантные фрагменты:",
        ]
        for index, chunk in enumerate(chunks[:5], start=1):
            lines.append(f"{index}. [{chunk.article_title}] {chunk.chunk_text}")
        lines.append("")
        lines.append("Сформулируй короткую выжимку на 1-2 абзаца только по этим фрагментам.")
        return "\n".join(lines)

    @staticmethod
    def _fallback_summary(chunks: list[ChunkSearchResult]) -> str:
        paragraphs: list[str] = []
        for chunk in chunks[:2]:
            text = chunk.chunk_text.strip()
            if text:
                paragraphs.append(text)
        return "\n\n".join(paragraphs) if paragraphs else "По выбранному запросу релевантные фрагменты не найдены."

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

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        seen: set[str] = set()
        tokens: list[str] = []
        for match in TOKEN_RE.finditer(text.lower()):
            token = match.group(0)
            if token in seen:
                continue
            seen.add(token)
            tokens.append(token)
        return tokens

    def _token_overlap_score(self, query_tokens: list[str], title: str, chunk_text: str) -> float:
        haystack = f"{title} {chunk_text}".lower()
        if not query_tokens:
            return 0.0
        matches = 0
        for token in query_tokens:
            if token in haystack:
                matches += 1
        return matches / len(query_tokens)

    def _has_topical_match(self, query_tokens: list[str], chunks: list[ChunkSearchResult]) -> bool:
        if not chunks or not query_tokens:
            return False
        for chunk in chunks[:3]:
            if self._token_overlap_score(query_tokens, chunk.article_title, chunk.chunk_text) > 0:
                return True
            if chunk.vector_score >= 0.72:
                return True
        return False

    @staticmethod
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
        return numerator / math.sqrt(left_norm * right_norm)
