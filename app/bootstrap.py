from __future__ import annotations

from dataclasses import dataclass
from sqlite3 import Connection

from app.config.settings import Settings
from app.repositories import (
    ArticleChunkRepository,
    ArticleRepository,
    NarrativeAnalysisRepository,
    NarrativeArticleAnalysisRepository,
    SourceRepository,
)
from app.services import (
    ChunkingService,
    EmbeddingIndexService,
    NarrativeMaterializationService,
    RAGService,
    build_cached_narrative_intelligence_pipeline,
    build_default_narrative_intelligence_pipeline,
    create_embedding_client,
    create_llm_client,
)


@dataclass(frozen=True, slots=True)
class AppServices:
    source_repository: SourceRepository
    article_repository: ArticleRepository
    article_chunk_repository: ArticleChunkRepository
    narrative_analysis_repository: NarrativeAnalysisRepository
    narrative_article_analysis_repository: NarrativeArticleAnalysisRepository
    chunking_service: ChunkingService
    embedding_index_service: EmbeddingIndexService
    rag_service: RAGService


def build_app_services(connection: Connection, settings: Settings) -> AppServices:
    source_repository = SourceRepository(connection)
    article_repository = ArticleRepository(connection)
    article_chunk_repository = ArticleChunkRepository(connection)
    narrative_analysis_repository = NarrativeAnalysisRepository(connection)
    narrative_article_analysis_repository = NarrativeArticleAnalysisRepository(connection)

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
        narrative_analysis_repository=narrative_analysis_repository,
        narrative_article_analysis_repository=narrative_article_analysis_repository,
        chunking_service=chunking_service,
        embedding_index_service=embedding_index_service,
        rag_service=rag_service,
    )


def build_narrative_intelligence_services(connection: Connection, settings: Settings):
    source_repository = SourceRepository(connection)
    article_repository = ArticleRepository(connection)
    article_chunk_repository = ArticleChunkRepository(connection)
    article_analysis_repository = NarrativeArticleAnalysisRepository(connection)
    llm_client = create_llm_client(settings)
    embedding_client = create_embedding_client(settings)
    if llm_client is None or embedding_client is None:
        raise RuntimeError("Narrative intelligence requires both LLM and embedding clients.")
    return build_cached_narrative_intelligence_pipeline(
        article_repository=article_repository,
        article_chunk_repository=article_chunk_repository,
        source_repository=source_repository,
        article_analysis_repository=article_analysis_repository,
        llm_client=llm_client,
        embedding_client=embedding_client,
    )


def build_narrative_materialization_service(connection: Connection) -> NarrativeMaterializationService:
    return NarrativeMaterializationService(NarrativeAnalysisRepository(connection))
