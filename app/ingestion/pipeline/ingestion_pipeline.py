from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from app.ingestion.discovery import RSSDiscoveryService, SectionPageDiscoveryService
from app.ingestion.fetcher import FetchError, HttpFetcher
from app.ingestion.parsers import get_article_parser
from app.ingestion.pipeline.validation import ParsedArticleValidationError, validate_parsed_article
from app.ingestion.sources import SourceConfig
from app.models import ArticleCreate, SourceCreate
from app.repositories import ArticleRepository, SourceRepository
from app.utils.text import estimate_word_count

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestionRunResult:
    discovered_urls: int
    fetched_urls: int
    parsed_articles: int
    saved_articles: int
    skipped_existing: int
    skipped_invalid: int
    failed_urls: int


class IngestionPipeline:
    def __init__(
        self,
        *,
        fetcher: HttpFetcher,
        source_repository: SourceRepository,
        article_repository: ArticleRepository,
        rss_discovery_service: RSSDiscoveryService | None = None,
        section_discovery_service: SectionPageDiscoveryService | None = None,
        min_body_length: int = 120,
    ) -> None:
        self.fetcher = fetcher
        self.source_repository = source_repository
        self.article_repository = article_repository
        self.rss_discovery_service = rss_discovery_service or RSSDiscoveryService(fetcher)
        self.section_discovery_service = section_discovery_service or SectionPageDiscoveryService(fetcher)
        self.min_body_length = min_body_length

    def run_once(self, source_config: SourceConfig, *, limit: int | None = None) -> IngestionRunResult:
        source = self._get_or_create_source(source_config)
        parser = get_article_parser(source_config.parser_type)
        candidate_urls = self._discover_urls(source_config, limit=limit)

        fetched_urls = 0
        parsed_articles = 0
        saved_articles = 0
        skipped_existing = 0
        skipped_invalid = 0
        failed_urls = 0

        for url in candidate_urls:
            if self.article_repository.get_article_by_url(url) is not None:
                skipped_existing += 1
                continue

            try:
                response = self.fetcher.fetch(url)
                fetched_urls += 1
                parsed_article = parser.parse(response.text, url)
                validate_parsed_article(parsed_article, min_body_length=self.min_body_length)
                parsed_articles += 1
            except ParsedArticleValidationError:
                skipped_invalid += 1
                logger.exception("Parsed article did not pass validation: %s", url)
                continue
            except (FetchError, ValueError):
                failed_urls += 1
                logger.exception("Failed to ingest article %s", url)
                continue

            article = ArticleCreate(
                source_id=source.id,
                url=url,
                title=parsed_article.title,
                subtitle=parsed_article.subtitle,
                body_text=parsed_article.body_text,
                published_at=parsed_article.published_at,
                author=parsed_article.author,
                category=parsed_article.category,
                language="ru",
                content_hash=self._compute_content_hash(parsed_article.body_text),
                word_count=estimate_word_count(parsed_article.body_text),
                is_canonical=True,
                duplicate_group_id=None,
            )

            try:
                self.article_repository.create_article(article)
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

    @staticmethod
    def _compute_content_hash(body_text: str) -> str:
        return hashlib.sha256(body_text.encode("utf-8")).hexdigest()
