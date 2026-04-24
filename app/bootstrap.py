from __future__ import annotations

from dataclasses import dataclass
from sqlite3 import Connection

from app.config.settings import Settings
from app.repositories import (
    ArticleChunkRepository,
    ArticleRepository,
    ClaimClusterRepository,
    ClaimRepository,
    NarrativeResultRepository,
    NarrativeRunRepository,
    SourceRepository,
)
from app.services import (
    ClaimGrouper,
    ClaimExtractor,
    ChunkingService,
    EmbeddingIndexService,
    NarrativeLabelingService,
    NarrativeRunService,
    NarrativeScorer,
    RAGService,
    create_embedding_client,
    create_llm_client,
)


@dataclass(frozen=True, slots=True)
class AppServices:
    source_repository: SourceRepository
    article_repository: ArticleRepository
    article_chunk_repository: ArticleChunkRepository
    claim_repository: ClaimRepository
    narrative_run_repository: NarrativeRunRepository
    claim_cluster_repository: ClaimClusterRepository
    narrative_result_repository: NarrativeResultRepository
    claim_extractor: ClaimExtractor
    chunking_service: ChunkingService
    embedding_index_service: EmbeddingIndexService
    rag_service: RAGService
    narrative_run_service: NarrativeRunService


def build_app_services(connection: Connection, settings: Settings) -> AppServices:
    source_repository = SourceRepository(connection)
    article_repository = ArticleRepository(connection)
    article_chunk_repository = ArticleChunkRepository(connection)
    claim_repository = ClaimRepository(connection)
    narrative_run_repository = NarrativeRunRepository(connection)
    claim_cluster_repository = ClaimClusterRepository(connection)
    narrative_result_repository = NarrativeResultRepository(connection)

    llm_client = create_llm_client(settings)
    embedding_client = create_embedding_client(settings)

    claim_extractor = ClaimExtractor(llm_client=llm_client)
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
    narrative_run_service = NarrativeRunService(
        article_repository=article_repository,
        claim_repository=claim_repository,
        narrative_run_repository=narrative_run_repository,
        claim_cluster_repository=claim_cluster_repository,
        narrative_result_repository=narrative_result_repository,
        claim_grouper=ClaimGrouper(),
        narrative_scorer=NarrativeScorer(),
        narrative_labeling_service=NarrativeLabelingService(llm_client=llm_client),
    )

    return AppServices(
        source_repository=source_repository,
        article_repository=article_repository,
        article_chunk_repository=article_chunk_repository,
        claim_repository=claim_repository,
        narrative_run_repository=narrative_run_repository,
        claim_cluster_repository=claim_cluster_repository,
        narrative_result_repository=narrative_result_repository,
        claim_extractor=claim_extractor,
        chunking_service=chunking_service,
        embedding_index_service=embedding_index_service,
        rag_service=rag_service,
        narrative_run_service=narrative_run_service,
    )
