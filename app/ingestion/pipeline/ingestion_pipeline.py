from __future__ import annotations

import logging
from dataclasses import dataclass

from app.ingestion.discovery import RSSDiscoveryService, SectionPageDiscoveryService
from app.ingestion.fetcher import FetchError, HttpFetcher
from app.ingestion.parsers import get_article_parser
from app.ingestion.pipeline.validation import (
    ParsedArticleValidationError,
    validate_normalized_article,
    validate_parsed_article,
)
from app.ingestion.sources import SourceConfig
from app.models import ArticleCreate, SourceCreate
from app.repositories import ArticleChunkRepository, ArticleRepository, ClaimRepository, SourceRepository
from app.services import (
    ArticleNormalizer,
    ClaimExtractor,
    ChunkingService,
    DeduplicationService,
    EmbeddingIndexService,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestionRunResult:
    discovered_urls: int
    fetched_urls: int
    parsed_articles: int
    saved_articles: int
    skipped_existing: int
    skipped_duplicates: int
    skipped_invalid: int
    failed_urls: int


class IngestionPipeline:
    def __init__(
        self,
        *,
        fetcher: HttpFetcher,
        source_repository: SourceRepository,
        article_repository: ArticleRepository,
        article_chunk_repository: ArticleChunkRepository | None = None,
        claim_repository: ClaimRepository | None = None,
        rss_discovery_service: RSSDiscoveryService | None = None,
        section_discovery_service: SectionPageDiscoveryService | None = None,
        article_normalizer: ArticleNormalizer | None = None,
        deduplication_service: DeduplicationService | None = None,
        chunking_service: ChunkingService | None = None,
        claim_extractor: ClaimExtractor | None = None,
        embedding_index_service: EmbeddingIndexService | None = None,
        min_body_length: int = 120,
        enable_chunking: bool = True,
        enable_claim_extraction: bool = True,
        enable_embeddings: bool = True,
    ) -> None:
        self.fetcher = fetcher
        self.source_repository = source_repository
        self.article_repository = article_repository
        self.article_chunk_repository = article_chunk_repository
        self.claim_repository = claim_repository
        self.rss_discovery_service = rss_discovery_service or RSSDiscoveryService(fetcher)
        self.section_discovery_service = section_discovery_service or SectionPageDiscoveryService(fetcher)
        self.article_normalizer = article_normalizer or ArticleNormalizer()
        self.deduplication_service = deduplication_service or DeduplicationService(article_repository)
        self.chunking_service = chunking_service or ChunkingService()
        self.claim_extractor = claim_extractor or ClaimExtractor()
        self.embedding_index_service = embedding_index_service
        self.min_body_length = min_body_length
        self.enable_chunking = enable_chunking
        self.enable_claim_extraction = enable_claim_extraction
        self.enable_embeddings = enable_embeddings

    def run_once(self, source_config: SourceConfig, *, limit: int | None = None) -> IngestionRunResult:
        candidate_urls = self._discover_urls(source_config, limit=limit)
        return self.run_urls(source_config, candidate_urls)

    def run_urls(self, source_config: SourceConfig, candidate_urls: list[str]) -> IngestionRunResult:
        source = self._get_or_create_source(source_config)
        parser = get_article_parser(source_config.parser_type)

        fetched_urls = 0
        parsed_articles = 0
        saved_articles = 0
        skipped_existing = 0
        skipped_duplicates = 0
        skipped_invalid = 0
        failed_urls = 0

        for url in candidate_urls:
            try:
                response = self.fetcher.fetch(url)
                fetched_urls += 1
                parsed_article = parser.parse(response.text, url)
                validate_parsed_article(parsed_article, min_body_length=self.min_body_length)
                normalized_article = self.article_normalizer.normalize(parsed_article, url=url)
                validate_normalized_article(
                    title=normalized_article.title,
                    body_text=normalized_article.body_text,
                )
                parsed_articles += 1
            except ParsedArticleValidationError:
                skipped_invalid += 1
                logger.exception("Parsed article did not pass validation: %s", url)
                continue
            except (FetchError, ValueError):
                failed_urls += 1
                logger.exception("Failed to ingest article %s", url)
                continue

            deduplication_result = self.deduplication_service.check_duplicate(normalized_article)
            if deduplication_result.is_duplicate:
                skipped_duplicates += 1
                if deduplication_result.duplicate_type == "url":
                    skipped_existing += 1
                if (
                    deduplication_result.canonical_article_id is not None
                    and deduplication_result.duplicate_group_id is not None
                ):
                    self.article_repository.create_duplicate_record(
                        duplicate_group_id=deduplication_result.duplicate_group_id,
                        article_id=deduplication_result.canonical_article_id,
                        duplicate_type=deduplication_result.duplicate_type or "unknown",
                        is_primary=True,
                        similarity_score=1.0,
                    )
                continue

            article = ArticleCreate(
                source_id=source.id,
                url=normalized_article.url,
                title=normalized_article.title,
                subtitle=normalized_article.subtitle,
                body_text=normalized_article.body_text,
                published_at=normalized_article.published_at,
                author=normalized_article.author,
                category=normalized_article.category,
                language="ru",
                content_hash=normalized_article.content_hash,
                word_count=normalized_article.word_count,
                is_canonical=True,
                duplicate_group_id=None,
            )

            try:
                saved_article = self.article_repository.create_article(article)
                if (
                    self.enable_chunking
                    and self.article_chunk_repository is not None
                    and saved_article.is_canonical
                ):
                    chunks = self.chunking_service.chunk_article(saved_article)
                    if chunks:
                        created_chunks = self.article_chunk_repository.create_many(chunks)
                        if (
                            self.enable_embeddings
                            and self.embedding_index_service is not None
                            and created_chunks
                        ):
                            self.embedding_index_service.index_chunks(created_chunks)
                if (
                    self.enable_claim_extraction
                    and self.claim_repository is not None
                    and saved_article.is_canonical
                ):
                    claims = self.claim_extractor.extract(saved_article)
                    if claims:
                        self.claim_repository.create_many(claims)
                saved_articles += 1
            except Exception:
                failed_urls += 1
                logger.exception("Failed to save parsed article %s", url)

        return IngestionRunResult(
            discovered_urls=len(candidate_urls),
            fetched_urls=fetched_urls,
            parsed_articles=parsed_articles,
            saved_articles=saved_articles,
            skipped_existing=skipped_existing,
            skipped_duplicates=skipped_duplicates,
            skipped_invalid=skipped_invalid,
            failed_urls=failed_urls,
        )

    def _discover_urls(self, source_config: SourceConfig, *, limit: int | None) -> list[str]:
        urls = self.rss_discovery_service.discover(source_config, limit=limit)
        if limit is None or len(urls) < limit:
            remaining = None if limit is None else max(limit - len(urls), 0)
            if remaining != 0:
                urls.extend(self.section_discovery_service.discover(source_config, limit=remaining))
        deduplicated: list[str] = []
        seen: set[str] = set()
        for url in urls:
            if url in seen:
                continue
            seen.add(url)
            deduplicated.append(url)
        return deduplicated[:limit] if limit is not None else deduplicated

    def _get_or_create_source(self, source_config: SourceConfig):
        existing = self.source_repository.get_by_domain(source_config.domain)
        if existing is not None:
            return existing
        return self.source_repository.create(
            SourceCreate(
                name=source_config.name,
                domain=source_config.domain,
                base_url=source_config.base_url,
                source_type="news_site",
                language="ru",
                is_active=True,
            )
        )
