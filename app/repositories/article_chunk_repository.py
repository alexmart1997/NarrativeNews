from __future__ import annotations

import sqlite3

from app.models import ArticleChunk, ArticleChunkCreate, ChunkSearchResult
from app.repositories.base import BaseRepository


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

    def search_chunks(
        self,
        query: str,
        date_from: str,
        date_to: str,
        limit: int = 10,
    ) -> list[ChunkSearchResult]:
        tokens = [token.strip() for token in query.split() if token.strip()]
        if not tokens:
            return []

        where_parts = []
        params: list[object] = [date_from, date_to]
        for token in tokens:
            where_parts.append("ac.chunk_text LIKE ?")
            params.append(f"%{token}%")

        query_sql = f"""
            SELECT
                ac.id AS chunk_id,
                ac.article_id AS article_id,
                ac.chunk_index AS chunk_index,
                ac.chunk_text AS chunk_text,
                a.published_at AS published_at,
                a.title AS article_title,
                ({' + '.join(['CASE WHEN ac.chunk_text LIKE ? THEN 1 ELSE 0 END' for _ in tokens])}) AS match_score
            FROM article_chunks ac
            INNER JOIN articles a ON a.id = ac.article_id
            WHERE a.is_canonical = 1
              AND a.published_at BETWEEN ? AND ?
              AND ({' OR '.join(where_parts)})
            ORDER BY match_score DESC, a.published_at DESC, ac.article_id ASC, ac.chunk_index ASC
            LIMIT ?
        """
        score_params = [f"%{token}%" for token in tokens]
        final_params = tuple(score_params + params + [limit])
        rows = self._fetch_all(query_sql, final_params)
        return [self._row_to_search_result(row) for row in rows]

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
    def _row_to_search_result(row: sqlite3.Row) -> ChunkSearchResult:
        return ChunkSearchResult(
            chunk_id=row["chunk_id"],
            article_id=row["article_id"],
            chunk_index=row["chunk_index"],
            chunk_text=row["chunk_text"],
            published_at=row["published_at"],
            article_title=row["article_title"],
            match_score=row["match_score"],
        )
