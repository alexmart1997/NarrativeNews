from __future__ import annotations

import logging

from app.models import ArticleChunk
from app.repositories import ArticleChunkRepository
from app.services.llm import BaseEmbeddingClient, EmbeddingError


logger = logging.getLogger(__name__)


class EmbeddingIndexService:
    def __init__(
        self,
        article_chunk_repository: ArticleChunkRepository,
        embedding_client: BaseEmbeddingClient | None,
    ) -> None:
        self.article_chunk_repository = article_chunk_repository
        self.embedding_client = embedding_client

    @property
    def is_enabled(self) -> bool:
        return self.embedding_client is not None

    def index_chunks(self, chunks: list[ArticleChunk]) -> int:
        if self.embedding_client is None or not chunks:
            return 0

        indexed = 0
        texts = [chunk.chunk_text for chunk in chunks]
        try:
            vectors = self.embedding_client.embed_texts(texts)
        except EmbeddingError:
            logger.exception("Failed to generate embeddings for new chunks.")
            return 0

        for chunk, vector in zip(chunks, vectors, strict=False):
            if not vector:
                continue
            self.article_chunk_repository.upsert_chunk_embedding(
                chunk_id=chunk.id,
                model_name=self.embedding_client.model_name,
                embedding=vector,
            )
            indexed += 1
        return indexed

    def index_missing_embeddings(self, limit: int = 100) -> int:
        if self.embedding_client is None:
            return 0
        chunks = self.article_chunk_repository.list_chunks_without_embeddings(
            model_name=self.embedding_client.model_name,
            limit=limit,
        )
        return self.index_chunks(chunks)
