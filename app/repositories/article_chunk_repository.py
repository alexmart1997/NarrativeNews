from __future__ import annotations

import json
import re
import sqlite3

from app.models import (
    ArticleChunk,
    ArticleChunkCreate,
    ArticleChunkEmbedding,
    ChunkSearchResult,
    EmbeddedChunkCandidate,
)
from app.repositories.base import BaseRepository, compact_datetime_sql, normalize_datetime_bound


TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{2,}")


class ArticleChunkRepository(BaseRepository):
    def create_many(self, payloads: list[ArticleChunkCreate]) -> list[ArticleChunk]:
        created: list[ArticleChunk] = []
        for payload in payloads:
            cursor = self.connection.execute(
                """
                INSERT INTO article_chunks (
                    article_id, chunk_index, chunk_text, char_start, char_end, token_count
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.article_id,
                    payload.chunk_index,
                    payload.chunk_text,
                    payload.char_start,
                    payload.char_end,
                    payload.token_count,
                ),
            )
            row = self._fetch_one("SELECT * FROM article_chunks WHERE id = ?", (cursor.lastrowid,))
            if row is not None:
                created.append(self._row_to_chunk(row))
        self.connection.commit()
        return created

    def list_by_article_id(self, article_id: int) -> list[ArticleChunk]:
        rows = self._fetch_all(
            """
            SELECT *
            FROM article_chunks
            WHERE article_id = ?
            ORDER BY chunk_index ASC
            """,
            (article_id,),
        )
        return [self._row_to_chunk(row) for row in rows]

    def list_chunks_without_embeddings(self, model_name: str, limit: int = 100) -> list[ArticleChunk]:
        rows = self._fetch_all(
            """
            SELECT ac.*
            FROM article_chunks ac
            LEFT JOIN article_chunk_embeddings ace
              ON ace.chunk_id = ac.id AND ace.model_name = ?
            WHERE ace.id IS NULL
            ORDER BY ac.id ASC
            LIMIT ?
            """,
            (model_name, limit),
        )
        return [self._row_to_chunk(row) for row in rows]

    def upsert_chunk_embedding(
        self,
        *,
        chunk_id: int,
        model_name: str,
        embedding: list[float],
    ) -> ArticleChunkEmbedding:
        payload = json.dumps(embedding, ensure_ascii=False)
        self.connection.execute(
            """
            INSERT INTO article_chunk_embeddings (chunk_id, model_name, embedding_json, dimension)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(chunk_id, model_name) DO UPDATE SET
                embedding_json = excluded.embedding_json,
                dimension = excluded.dimension,
                created_at = CURRENT_TIMESTAMP
            """,
            (chunk_id, model_name, payload, len(embedding)),
        )
        self.connection.commit()
        row = self._fetch_one(
            """
            SELECT *
            FROM article_chunk_embeddings
            WHERE chunk_id = ? AND model_name = ?
            """,
            (chunk_id, model_name),
        )
        if row is None:
            raise RuntimeError("Failed to load saved chunk embedding.")
        return self._row_to_embedding(row)

    def search_chunks(
        self,
        query: str,
        date_from: str,
        date_to: str,
        limit: int = 10,
        source_domains: list[str] | None = None,
    ) -> list[ChunkSearchResult]:
        return self.search_chunks_lexical(query, date_from, date_to, limit, source_domains=source_domains)

    def search_chunks_lexical(
        self,
        query: str,
        date_from: str,
        date_to: str,
        limit: int = 20,
        source_domains: list[str] | None = None,
    ) -> list[ChunkSearchResult]:
        date_from = normalize_datetime_bound(date_from) or date_from
        date_to = normalize_datetime_bound(date_to) or date_to
        tokens = self._tokenize(query)
        if not tokens:
            return []

        match_query = " OR ".join(f"{token}*" for token in tokens)
        source_clause = ""
        params: list[object] = [match_query, date_from, date_to]
        if source_domains:
            placeholders = ", ".join("?" for _ in source_domains)
            source_clause = f" AND s.domain IN ({placeholders})"
            params.extend(source_domains)
        params.append(limit)
        try:
            rows = self._fetch_all(
                f"""
                SELECT
                    ac.id AS chunk_id,
                    ac.article_id AS article_id,
                    ac.chunk_index AS chunk_index,
                    ac.chunk_text AS chunk_text,
                    a.published_at AS published_at,
                    a.title AS article_title,
                    bm25(article_chunks_fts) AS match_score
                FROM article_chunks_fts
                INNER JOIN article_chunks ac ON ac.id = article_chunks_fts.rowid
                INNER JOIN articles a ON a.id = ac.article_id
                INNER JOIN sources s ON s.id = a.source_id
                WHERE article_chunks_fts MATCH ?
                  AND a.is_canonical = 1
                  AND {compact_datetime_sql("a.published_at")} BETWEEN ? AND ?
                """
                + source_clause
                + f"""
                ORDER BY match_score ASC, {compact_datetime_sql("a.published_at")} DESC, ac.article_id ASC, ac.chunk_index ASC
                LIMIT ?
                """,
                tuple(params),
            )
            return [self._row_to_search_result(row) for row in rows]
        except sqlite3.OperationalError:
            return self._search_chunks_like(query, date_from, date_to, limit, source_domains=source_domains)

    def list_vector_candidates(
        self,
        *,
        model_name: str,
        date_from: str,
        date_to: str,
        source_domains: list[str] | None = None,
    ) -> list[EmbeddedChunkCandidate]:
        date_from = normalize_datetime_bound(date_from) or date_from
        date_to = normalize_datetime_bound(date_to) or date_to
        source_clause = ""
        params: list[object] = [model_name, date_from, date_to]
        if source_domains:
            placeholders = ", ".join("?" for _ in source_domains)
            source_clause = f" AND s.domain IN ({placeholders})"
            params.extend(source_domains)
        rows = self._fetch_all(
            f"""
            SELECT
                ac.id AS chunk_id,
                ac.article_id AS article_id,
                ac.chunk_index AS chunk_index,
                ac.chunk_text AS chunk_text,
                a.published_at AS published_at,
                a.title AS article_title,
                ace.embedding_json AS embedding_json
            FROM article_chunk_embeddings ace
            INNER JOIN article_chunks ac ON ac.id = ace.chunk_id
            INNER JOIN articles a ON a.id = ac.article_id
            INNER JOIN sources s ON s.id = a.source_id
            WHERE ace.model_name = ?
              AND a.is_canonical = 1
              AND {compact_datetime_sql("a.published_at")} BETWEEN ? AND ?
            """
            + source_clause
            + f"""
            ORDER BY {compact_datetime_sql("a.published_at")} DESC, ac.article_id ASC, ac.chunk_index ASC
            """,
            tuple(params),
        )

        candidates: list[EmbeddedChunkCandidate] = []
        for row in rows:
            try:
                embedding = json.loads(row["embedding_json"])
            except json.JSONDecodeError:
                continue
            if not isinstance(embedding, list) or not embedding:
                continue
            try:
                vector = [float(value) for value in embedding]
            except (TypeError, ValueError):
                continue
            candidates.append(
                EmbeddedChunkCandidate(
                    chunk_id=row["chunk_id"],
                    article_id=row["article_id"],
                    chunk_index=row["chunk_index"],
                    chunk_text=row["chunk_text"],
                    published_at=row["published_at"],
                    article_title=row["article_title"],
                    embedding=vector,
                )
            )
        return candidates

    def _search_chunks_like(
        self,
        query: str,
        date_from: str,
        date_to: str,
        limit: int,
        source_domains: list[str] | None = None,
    ) -> list[ChunkSearchResult]:
        date_from = normalize_datetime_bound(date_from) or date_from
        date_to = normalize_datetime_bound(date_to) or date_to
        tokens = self._tokenize(query)
        if not tokens:
            return []

        where_parts = []
        params: list[object] = []
        score_parts: list[str] = []
        for token in tokens:
            where_parts.append("(ac.chunk_text LIKE ? OR a.title LIKE ?)")
            params.extend([f"%{token}%", f"%{token}%"])
            score_parts.append("CASE WHEN ac.chunk_text LIKE ? OR a.title LIKE ? THEN 1 ELSE 0 END")

        score_params: list[object] = []
        for token in tokens:
            score_params.extend([f"%{token}%", f"%{token}%"])

        source_clause = ""
        source_params: list[object] = []
        if source_domains:
            placeholders = ", ".join("?" for _ in source_domains)
            source_clause = f" AND s.domain IN ({placeholders})"
            source_params.extend(source_domains)

        rows = self._fetch_all(
            f"""
            SELECT
                ac.id AS chunk_id,
                ac.article_id AS article_id,
                ac.chunk_index AS chunk_index,
                ac.chunk_text AS chunk_text,
                a.published_at AS published_at,
                a.title AS article_title,
                ({' + '.join(score_parts)}) AS match_score
            FROM article_chunks ac
            INNER JOIN articles a ON a.id = ac.article_id
            INNER JOIN sources s ON s.id = a.source_id
            WHERE a.is_canonical = 1
              AND {compact_datetime_sql("a.published_at")} BETWEEN ? AND ?
              {source_clause}
              AND ({' OR '.join(where_parts)})
            ORDER BY match_score DESC, {compact_datetime_sql("a.published_at")} DESC, ac.article_id ASC, ac.chunk_index ASC
            LIMIT ?
            """,
            tuple(score_params + [date_from, date_to] + source_params + params + [limit]),
        )
        return [self._row_to_search_result(row) for row in rows]

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

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> ArticleChunk:
        return ArticleChunk(
            id=row["id"],
            article_id=row["article_id"],
            chunk_index=row["chunk_index"],
            chunk_text=row["chunk_text"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            token_count=row["token_count"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_embedding(row: sqlite3.Row) -> ArticleChunkEmbedding:
        return ArticleChunkEmbedding(
            id=row["id"],
            chunk_id=row["chunk_id"],
            model_name=row["model_name"],
            embedding=json.loads(row["embedding_json"]),
            dimension=row["dimension"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_search_result(row: sqlite3.Row) -> ChunkSearchResult:
        return ChunkSearchResult(
            chunk_id=row["chunk_id"],
            article_id=row["article_id"],
            chunk_index=row["chunk_index"],
            chunk_text=row["chunk_text"],
            published_at=row["published_at"],
            article_title=row["article_title"],
            match_score=float(row["match_score"] or 0.0),
        )
