from __future__ import annotations

from dataclasses import dataclass
from sqlite3 import Connection

from app.config.settings import Settings
from app.repositories import ArticleChunkRepository, ArticleRepository, SourceRepository
from app.services import ChunkingService, EmbeddingIndexService, RAGService, create_embedding_client, create_llm_client


@dataclass(frozen=True, slots=True)
class AppServices:
    source_repository: SourceRepository
    article_repository: ArticleRepository
    article_chunk_repository: ArticleChunkRepository
    chunking_service: ChunkingService
    embedding_index_service: EmbeddingIndexService
    rag_service: RAGService


def build_app_services(connection: Connection, settings: Settings) -> AppServices:
    source_repository = SourceRepository(connection)
    article_repository = ArticleRepository(connection)
    article_chunk_repository = ArticleChunkRepository(connection)

    llm_client = create_llm_client(settings)
    embedding_client = create_embedding_client(settings)

    chunking_service = ChunkingService()
    embedding_index_service = EmbeddingIndexService(
        article_chunk_repository=article_chunk_repository,
        embedding_client=embedding_client,
    )
    rag_service = RAGService(
        article_chunk_repository=article_chunk_repository,
        article_repository=article_repository,
        llm_client=llm_client,
        embedding_client=embedding_client,
        hybrid_limit=settings.rag_hybrid_limit,
        rerank_limit=settings.rag_rerank_limit,
    )

    return AppServices(
        source_repository=source_repository,
        article_repository=article_repository,
        article_chunk_repository=article_chunk_repository,
        chunking_service=chunking_service,
        embedding_index_service=embedding_index_service,
        rag_service=rag_service,
    )
