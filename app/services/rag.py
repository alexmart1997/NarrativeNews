from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re

from app.models import Article, ChunkSearchResult, RAGAnswerResult
from app.repositories import ArticleChunkRepository, ArticleRepository
from app.services.llm import BaseEmbeddingClient, BaseLLMClient
from app.services.reranker import BaseChunkReranker


TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{2,}")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
STOPWORDS = {
    "что",
    "как",
    "где",
    "когда",
    "это",
    "эта",
    "этот",
    "эти",
    "про",
    "для",
    "при",
    "или",
    "его",
    "ее",
    "её",
    "она",
    "они",
    "оно",
    "есть",
    "был",
    "была",
    "были",
    "так",
    "еще",
    "ещё",
    "уже",
    "над",
    "под",
    "без",
    "после",
    "перед",
    "между",
    "если",
    "чем",
    "ли",
    "же",
    "по",
    "на",
    "в",
    "во",
    "из",
    "к",
    "ко",
    "с",
    "со",
    "о",
    "об",
    "от",
    "до",
    "за",
    "не",
    "а",
    "и",
    "но",
}
BROAD_TERMS = {
    "россия",
    "россии",
    "россию",
    "россий",
    "страна",
    "страны",
    "сша",
    "европа",
    "европы",
    "москва",
    "москвы",
    "украина",
    "украины",
    "украину",
}
ACTION_TERMS = {
    "????????????????????????",
    "????????????????????????????????",
    "?????????????????????????",
    "??????????????????????????????????",
    "?????????????????????????????",
    "?????????????????????????????",
    "?????????????????????????????????",
    "?????????????????????????????????????????",
    "????????????????",
    "?????????????????????",
    "?????????????????????",
    "????????????????????????",
    "block",
    "blocking",
    "ban",
    "banned",
    "restrict",
    "restriction",
    "restrictions",
    "arrest",
}
RUSSIAN_ENDINGS = (
    "иями",
    "ями",
    "ами",
    "его",
    "ого",
    "ему",
    "ому",
    "иях",
    "ах",
    "ях",
    "ия",
    "ья",
    "ие",
    "ые",
    "ой",
    "ий",
    "ый",
    "ая",
    "яя",
    "ое",
    "ее",
    "ам",
    "ям",
    "ом",
    "ем",
    "ов",
    "ев",
    "ей",
    "ию",
    "ью",
    "ых",
    "их",
    "ую",
    "юю",
    "а",
    "я",
    "ы",
    "и",
    "е",
    "у",
    "ю",
    "о",
)
NOISE_PATTERNS = (
    re.compile(r"ваш браузер не поддерживает", re.IGNORECASE),
    re.compile(r"данный формат видео", re.IGNORECASE),
    re.compile(r"смотрите видео", re.IGNORECASE),
    re.compile(r"подпишит", re.IGNORECASE),
    re.compile(r"ria\.ru/\d+", re.IGNORECASE),
)


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
        reranker: BaseChunkReranker | None = None,
        hybrid_limit: int = 24,
        rerank_limit: int = 8,
    ) -> None:
        self.article_chunk_repository = article_chunk_repository
        self.article_repository = article_repository
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.reranker = reranker
        self.hybrid_limit = hybrid_limit
        self.rerank_limit = rerank_limit

    def search_chunks(
        self,
        query: str,
        date_from: str,
        date_to: str,
        limit: int = 10,
        source_domains: list[str] | None = None,
    ) -> list[ChunkSearchResult]:
        query_terms = self._extract_query_terms(query)
        if not query_terms:
            return []

        candidates = self._hybrid_retrieve(
            query_terms=query_terms,
            date_from=date_from,
            date_to=date_to,
            limit=max(limit * 3, self.hybrid_limit),
            source_domains=source_domains,
        )
        reranked = self._rerank(query, candidates, limit=max(limit * 2, self.rerank_limit))
        filtered = self._filter_topical_chunks(query_terms, reranked)
        diversified = self._diversify_chunks(filtered, max_per_article=2)
        return diversified[:limit]

    def search(
        self,
        query: str,
        date_from: str,
        date_to: str,
        limit: int = 10,
        source_domains: list[str] | None = None,
    ) -> RAGSearchResult:
        chunks = self.search_chunks(query, date_from, date_to, limit, source_domains=source_domains)
        articles = self._select_source_articles(chunks, max_articles=5)
        return RAGSearchResult(chunks=chunks, articles=articles)

    def answer(
        self,
        query: str,
        date_from: str,
        date_to: str,
        limit: int = 10,
        include_debug_chunks: bool = False,
        source_domains: list[str] | None = None,
    ) -> RAGAnswerResult:
        chunks = self.search_chunks(query, date_from, date_to, limit, source_domains=source_domains)
        source_articles = self._select_source_articles(chunks, max_articles=5)
        summary_text = self._generate_summary(query, chunks)
        return RAGAnswerResult(
            summary_text=summary_text,
            source_articles=source_articles,
            top_chunks=chunks if include_debug_chunks else None,
        )

    def _hybrid_retrieve(
        self,
        *,
        query_terms: list[dict[str, str]],
        date_from: str,
        date_to: str,
        limit: int,
        source_domains: list[str] | None = None,
    ) -> list[ChunkSearchResult]:
        lexical_query = self._build_lexical_query(query_terms)
        lexical_rows = self.article_chunk_repository.search_chunks_lexical(
            lexical_query,
            date_from,
            date_to,
            limit=limit,
            source_domains=source_domains,
        )

        anchor_terms = self._anchor_terms(query_terms)
        merged: dict[int, ChunkSearchResult] = {}
        for rank, row in enumerate(lexical_rows, start=1):
            lexical_score = 1.0 / rank
            overlap_score = self._token_overlap_score(query_terms, row.article_title, row.chunk_text)
            anchor_score = self._anchor_overlap_score(anchor_terms, row.article_title, row.chunk_text)
            quality_score = self._chunk_quality_score(row.chunk_text)
            final_score = (
                0.48 * lexical_score
                + 0.22 * overlap_score
                + 0.20 * anchor_score
                + quality_score
            )
            merged[row.chunk_id] = replace(
                row,
                lexical_score=lexical_score,
                final_score=final_score,
            )

        vector_rows: list[ChunkSearchResult] = []
        if self.embedding_client is not None:
            try:
                query_embedding = self.embedding_client.embed_text(" ".join(term["raw"] for term in query_terms))
            except Exception:
                query_embedding = []
            if query_embedding:
                vector_rows = self._search_vector_candidates(
                    query_embedding=query_embedding,
                    query_terms=query_terms,
                    anchor_terms=anchor_terms,
                    date_from=date_from,
                    date_to=date_to,
                    limit=limit,
                    source_domains=source_domains,
                )

        for rank, row in enumerate(vector_rows, start=1):
            vector_rank_bonus = 1.0 / rank
            overlap_score = self._token_overlap_score(query_terms, row.article_title, row.chunk_text)
            anchor_score = self._anchor_overlap_score(anchor_terms, row.article_title, row.chunk_text)
            existing = merged.get(row.chunk_id)
            if existing is None:
                final_score = (
                    0.45 * row.vector_score
                    + 0.12 * vector_rank_bonus
                    + 0.18 * overlap_score
                    + 0.18 * anchor_score
                    + self._chunk_quality_score(row.chunk_text)
                )
                merged[row.chunk_id] = replace(row, final_score=final_score)
                continue

            merged[row.chunk_id] = replace(
                existing,
                vector_score=max(existing.vector_score, row.vector_score),
                final_score=(
                    existing.final_score
                    + 0.22 * row.vector_score
                    + 0.05 * vector_rank_bonus
                    + 0.06 * anchor_score
                    + 0.03 * overlap_score
                ),
            )

        ranked = sorted(
            merged.values(),
            key=lambda item: (
                item.final_score,
                item.rerank_score,
                item.lexical_score,
                item.vector_score,
                item.published_at,
            ),
            reverse=True,
        )
        return ranked[:limit]

    def _search_vector_candidates(
        self,
        *,
        query_embedding: list[float],
        query_terms: list[dict[str, str]],
        anchor_terms: set[str],
        date_from: str,
        date_to: str,
        limit: int,
        source_domains: list[str] | None = None,
    ) -> list[ChunkSearchResult]:
        if self.embedding_client is None:
            return []

        candidates = self.article_chunk_repository.list_vector_candidates(
            model_name=self.embedding_client.model_name,
            date_from=date_from,
            date_to=date_to,
            source_domains=source_domains,
        )
        scored: list[ChunkSearchResult] = []
        for candidate in candidates:
            similarity = self._cosine_similarity(query_embedding, candidate.embedding)
            if similarity <= 0:
                continue
            overlap_score = self._token_overlap_score(query_terms, candidate.article_title, candidate.chunk_text)
            anchor_score = self._anchor_overlap_score(anchor_terms, candidate.article_title, candidate.chunk_text)
            final_score = (
                0.54 * similarity
                + 0.16 * overlap_score
                + 0.16 * anchor_score
                + self._chunk_quality_score(candidate.chunk_text)
            )
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

        initial = candidates[:limit]
        reranked = self._model_rerank(query, initial)
        reranked = self._llm_listwise_rerank(query, reranked)
        return reranked[:limit]

    def _model_rerank(
        self,
        query: str,
        candidates: list[ChunkSearchResult],
    ) -> list[ChunkSearchResult]:
        if not candidates or self.reranker is None:
            return candidates

        try:
            scores = self.reranker.score(query, candidates)
        except Exception:
            return candidates
        if len(scores) != len(candidates):
            return candidates

        reranked: list[ChunkSearchResult] = []
        for candidate, score in zip(candidates, scores, strict=False):
            reranked.append(
                replace(
                    candidate,
                    model_rerank_score=score,
                    final_score=candidate.final_score + (0.42 * score),
                )
            )
        reranked.sort(
            key=lambda item: (
                item.model_rerank_score,
                item.final_score,
                item.lexical_score,
                item.vector_score,
                item.published_at,
            ),
            reverse=True,
        )
        return reranked

    def _llm_listwise_rerank(
        self,
        query: str,
        candidates: list[ChunkSearchResult],
    ) -> list[ChunkSearchResult]:
        initial = candidates
        if self.llm_client is None:
            return initial

        prompt_lines = [
            f"Query: {query}",
            "",
            'Return JSON only in the form {"ranked_chunk_ids": [id1, id2, ...]}.',
            "Rank the chunks from most useful to least useful for answering the query.",
            "Prefer direct topical relevance, factual density, and non-boilerplate text.",
            "",
            "Chunks:",
        ]
        for item in initial:
            prompt_lines.append(f"- id={item.chunk_id} | title={item.article_title} | text={item.chunk_text}")
        prompt = "\n".join(prompt_lines)
        system_prompt = (
            "You are reranking retrieved news chunks for a Russian-language RAG system. "
            "Return valid JSON only. Do not invent chunk ids."
        )
        try:
            payload = self.llm_client.generate_json(
                prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=180,
            )
        except Exception:
            return initial

        ranked_ids = payload.get("ranked_chunk_ids", [])
        if not isinstance(ranked_ids, list):
            return initial

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
                    final_score=item.final_score + (0.30 / index),
                )
            )

        if not reranked:
            return initial

        for item in initial:
            if item.chunk_id not in used:
                reranked.append(item)
        reranked.sort(
            key=lambda item: (
                item.rerank_score,
                item.model_rerank_score,
                item.final_score,
                item.lexical_score,
                item.vector_score,
            ),
            reverse=True,
        )
        return reranked

    def _filter_topical_chunks(
        self,
        query_terms: list[dict[str, str]],
        chunks: list[ChunkSearchResult],
    ) -> list[ChunkSearchResult]:
        if not chunks:
            return []

        anchor_terms = self._anchor_terms(query_terms)
        is_narrow_query = self._is_narrow_query(query_terms)
        required_anchor_matches = 2 if is_narrow_query and len(anchor_terms) >= 2 else 1
        strict_matches: list[ChunkSearchResult] = []
        semantic_matches: list[ChunkSearchResult] = []
        fallback_matches: list[ChunkSearchResult] = []
        strict_ids: set[int] = set()
        anchor_present = False

        for chunk in chunks:
            quality_score = self._chunk_quality_score(chunk.chunk_text)
            if quality_score <= -0.45:
                continue

            overlap = self._token_overlap_score(query_terms, chunk.article_title, chunk.chunk_text)
            anchor_score = self._anchor_overlap_score(anchor_terms, chunk.article_title, chunk.chunk_text)
            anchor_match_count = self._anchor_match_count(anchor_terms, chunk.article_title, chunk.chunk_text)
            if anchor_score > 0:
                anchor_present = True

            if is_narrow_query and anchor_terms:
                if anchor_match_count >= required_anchor_matches and (overlap >= 0.40 or chunk.vector_score >= 0.55):
                    strict_matches.append(chunk)
                    strict_ids.add(chunk.chunk_id)
                    continue
                if anchor_match_count >= required_anchor_matches and chunk.vector_score >= 0.72:
                    semantic_matches.append(chunk)
                    continue
                continue

            if overlap >= 0.50 or anchor_score >= 0.50:
                strict_matches.append(chunk)
                strict_ids.add(chunk.chunk_id)
                continue
            if anchor_score > 0 and chunk.vector_score >= 0.72:
                semantic_matches.append(chunk)
                continue
            if overlap > 0 and chunk.vector_score >= 0.66:
                semantic_matches.append(chunk)
                continue
            if chunk.vector_score >= 0.82 and chunk.final_score >= 0.50:
                semantic_matches.append(chunk)
                continue
            if chunk.final_score >= 0.62 and not self._is_low_value_chunk(chunk.chunk_text):
                fallback_matches.append(chunk)

        if strict_matches:
            return strict_matches + [item for item in semantic_matches if item.chunk_id not in strict_ids]
        if is_narrow_query:
            return []
        if anchor_terms and anchor_present:
            return semantic_matches
        if semantic_matches:
            return semantic_matches
        return fallback_matches or chunks

    def _generate_summary(self, query: str, chunks: list[ChunkSearchResult]) -> str:
        if not chunks:
            return "По выбранному запросу релевантные фрагменты не найдены."

        useful_chunks = [chunk for chunk in chunks if not self._is_low_value_chunk(chunk.chunk_text)]
        if not useful_chunks:
            useful_chunks = chunks

        if self.llm_client is None:
            return self._fallback_summary(useful_chunks)

        prompt = self._build_prompt(query, useful_chunks[:5])
        system_prompt = (
            "You summarize Russian news retrieval results. "
            "Answer in Russian. Use only the supplied fragments. "
            "Do not invent facts or add outside context."
        )
        try:
            text = self.llm_client.generate_text(
                prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=260,
            )
        except Exception:
            return self._fallback_summary(useful_chunks)

        cleaned = text.strip()
        if not cleaned or not CYRILLIC_RE.search(cleaned):
            return self._fallback_summary(useful_chunks)
        return cleaned

    @staticmethod
    def _build_prompt(query: str, chunks: list[ChunkSearchResult]) -> str:
        lines = [
            f"Запрос: {query}",
            "",
            "Релевантные фрагменты:",
        ]
        for index, chunk in enumerate(chunks, start=1):
            lines.append(f"{index}. [{chunk.article_title}] {chunk.chunk_text}")
        lines.append("")
        lines.append("Сделай короткую сводку на русском языке в 1-2 абзаца.")
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
        article_scores: dict[int, float] = {}
        for chunk in chunks:
            article_scores.setdefault(chunk.article_id, 0.0)
            article_scores[chunk.article_id] += max(chunk.final_score, 0.0) + (0.10 / (chunk.chunk_index + 1))

        ranked_article_ids = [
            article_id
            for article_id, _score in sorted(article_scores.items(), key=lambda item: item[1], reverse=True)[
                :max_articles
            ]
        ]

        articles: list[Article] = []
        for article_id in ranked_article_ids:
            article = self.article_repository.get_article_by_id(article_id)
            if article is not None:
                articles.append(article)
        return articles

    def _extract_query_terms(self, query: str) -> list[dict[str, str]]:
        terms: list[dict[str, str]] = []
        seen: set[str] = set()
        for match in TOKEN_RE.finditer(query.lower()):
            raw = match.group(0)
            if raw in STOPWORDS:
                continue
            normalized = self._normalize_token(raw)
            if normalized in STOPWORDS or len(normalized) < 2:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            terms.append({"raw": raw, "normalized": normalized})
        return terms

    def _build_lexical_query(self, query_terms: list[dict[str, str]]) -> str:
        parts: list[str] = []
        seen: set[str] = set()
        for term in query_terms:
            for value in (term["raw"], term["normalized"]):
                if len(value) < 3 or value in seen:
                    continue
                seen.add(value)
                parts.append(value)
        return " ".join(parts)

    def _token_overlap_score(self, query_terms: list[dict[str, str]], title: str, chunk_text: str) -> float:
        haystack_terms = self._extract_query_terms(f"{title} {chunk_text}")
        haystack_values = {term["raw"] for term in haystack_terms} | {term["normalized"] for term in haystack_terms}
        if not query_terms:
            return 0.0

        matches = 0
        for term in query_terms:
            if term["raw"] in haystack_values or term["normalized"] in haystack_values:
                matches += 1
        return matches / len(query_terms)

    def _anchor_terms(self, query_terms: list[dict[str, str]]) -> set[str]:
        meaningful = [
            term["normalized"]
            for term in query_terms
            if len(term["normalized"]) >= 5 and term["normalized"] not in BROAD_TERMS
        ]
        if meaningful:
            return set(meaningful)
        if len(query_terms) >= 2:
            ordered = sorted((term["normalized"] for term in query_terms), key=len, reverse=True)
            return set(ordered[:1])
        return set()

    def _anchor_overlap_score(self, anchor_terms: set[str], title: str, chunk_text: str) -> float:
        if not anchor_terms:
            return 0.0
        haystack_terms = self._extract_query_terms(f"{title} {chunk_text}")
        haystack_values = {term["raw"] for term in haystack_terms} | {term["normalized"] for term in haystack_terms}
        matches = sum(1 for term in anchor_terms if term in haystack_values)
        return matches / len(anchor_terms)

    def _anchor_match_count(self, anchor_terms: set[str], title: str, chunk_text: str) -> int:
        if not anchor_terms:
            return 0
        haystack_terms = self._extract_query_terms(f"{title} {chunk_text}")
        haystack_values = {term["raw"] for term in haystack_terms} | {term["normalized"] for term in haystack_terms}
        return sum(1 for term in anchor_terms if term in haystack_values)

    def _is_narrow_query(self, query_terms: list[dict[str, str]]) -> bool:
        if len(query_terms) < 2:
            return False
        anchor_terms = self._anchor_terms(query_terms)
        if len(anchor_terms) < 2:
            return False
        return any(term["normalized"] in ACTION_TERMS for term in query_terms)

    def _diversify_chunks(self, chunks: list[ChunkSearchResult], *, max_per_article: int) -> list[ChunkSearchResult]:
        diversified: list[ChunkSearchResult] = []
        per_article: dict[int, int] = {}
        for chunk in sorted(
            chunks,
            key=lambda item: (item.final_score, item.rerank_score, item.lexical_score, item.vector_score),
            reverse=True,
        ):
            count = per_article.get(chunk.article_id, 0)
            if count >= max_per_article:
                continue
            per_article[chunk.article_id] = count + 1
            diversified.append(chunk)
        return diversified

    def _chunk_quality_score(self, chunk_text: str) -> float:
        text = chunk_text.strip()
        if not text:
            return -0.80
        lowered = text.lower()
        if any(pattern.search(lowered) for pattern in NOISE_PATTERNS):
            return -0.65

        token_count = len(TOKEN_RE.findall(lowered))
        if token_count < 8:
            return -0.28
        if token_count < 18:
            return -0.08
        if token_count > 260:
            return -0.05
        return 0.06

    def _is_low_value_chunk(self, chunk_text: str) -> bool:
        return self._chunk_quality_score(chunk_text) <= -0.45

    @staticmethod
    def _normalize_token(token: str) -> str:
        normalized = token.lower()
        for ending in RUSSIAN_ENDINGS:
            if normalized.endswith(ending) and len(normalized) - len(ending) >= 4:
                return normalized[: -len(ending)]
        return normalized

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
