from __future__ import annotations

from dataclasses import dataclass

from app.models import Article, ArticleChunkCreate
from app.utils.text import estimate_word_count


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    target_chunk_size: int = 700
    min_chunk_size: int = 300


class ChunkingService:
    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk_article(self, article: Article) -> list[ArticleChunkCreate]:
        paragraphs = [paragraph.strip() for paragraph in article.body_text.split("\n\n") if paragraph.strip()]
        if not paragraphs:
            return []

        chunks: list[ArticleChunkCreate] = []
        current_parts: list[str] = []
        current_start: int | None = None
        current_end: int | None = None
        current_length = 0
        cursor = 0

        for paragraph in paragraphs:
            paragraph_start = article.body_text.find(paragraph, cursor)
            if paragraph_start == -1:
                paragraph_start = cursor
            paragraph_end = paragraph_start + len(paragraph)
            cursor = paragraph_end

            projected_length = current_length + len(paragraph) + (2 if current_parts else 0)
            if current_parts and projected_length > self.config.target_chunk_size and current_length >= self.config.min_chunk_size:
                chunks.append(
                    self._build_chunk(
                        article_id=article.id,
                        chunk_index=len(chunks),
                        parts=current_parts,
                        char_start=current_start,
                        char_end=current_end,
                    )
                )
                current_parts = []
                current_start = None
                current_end = None
                current_length = 0

            if current_start is None:
                current_start = paragraph_start
            current_end = paragraph_end
            current_parts.append(paragraph)
            current_length += len(paragraph) + (2 if len(current_parts) > 1 else 0)

        if current_parts:
            chunks.append(
                self._build_chunk(
                    article_id=article.id,
                    chunk_index=len(chunks),
                    parts=current_parts,
                    char_start=current_start,
                    char_end=current_end,
                )
            )

        return chunks

    @staticmethod
    def _build_chunk(
        *,
        article_id: int,
        chunk_index: int,
        parts: list[str],
        char_start: int | None,
        char_end: int | None,
    ) -> ArticleChunkCreate:
        chunk_text = "\n\n".join(parts).strip()
        return ArticleChunkCreate(
            article_id=article_id,
            chunk_index=chunk_index,
            chunk_text=chunk_text,
            char_start=char_start,
            char_end=char_end,
            token_count=estimate_word_count(chunk_text),
        )
