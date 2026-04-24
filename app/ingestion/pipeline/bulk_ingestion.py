from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from app.ingestion.discovery import ArchiveDiscoveryService
from app.ingestion.pipeline.ingestion_pipeline import IngestionPipeline, IngestionRunResult
from app.ingestion.sources import SourceConfig


@dataclass(frozen=True, slots=True)
class BulkIngestionResult:
    source_name: str
    date_from: date
    date_to: date
    days_processed: int
    discovered_urls: int
    fetched_urls: int
    parsed_articles: int
    saved_articles: int
    skipped_existing: int
    skipped_duplicates: int
    skipped_invalid: int
    failed_urls: int


class BulkIngestionService:
    def __init__(
        self,
        *,
        pipeline: IngestionPipeline,
        archive_discovery_service: ArchiveDiscoveryService,
    ) -> None:
        self.pipeline = pipeline
        self.archive_discovery_service = archive_discovery_service

    def run_for_date_range(
        self,
        source_config: SourceConfig,
        *,
        date_from: date,
        date_to: date,
        per_day_limit: int | None = None,
    ) -> BulkIngestionResult:
        discovery_by_day = self.archive_discovery_service.discover_for_date_range(
            source_config,
            date_from=date_from,
            date_to=date_to,
            per_day_limit=per_day_limit,
        )

        totals = {
            "discovered_urls": 0,
            "fetched_urls": 0,
            "parsed_articles": 0,
            "saved_articles": 0,
            "skipped_existing": 0,
            "skipped_duplicates": 0,
            "skipped_invalid": 0,
            "failed_urls": 0,
        }

        for urls in discovery_by_day.values():
            if not urls:
                continue
            run_result = self.pipeline.run_urls(source_config, urls)
            self._accumulate(totals, run_result)

        return BulkIngestionResult(
            source_name=source_config.name,
            date_from=date_from,
            date_to=date_to,
            days_processed=len(discovery_by_day),
            discovered_urls=totals["discovered_urls"],
            fetched_urls=totals["fetched_urls"],
            parsed_articles=totals["parsed_articles"],
            saved_articles=totals["saved_articles"],
            skipped_existing=totals["skipped_existing"],
            skipped_duplicates=totals["skipped_duplicates"],
            skipped_invalid=totals["skipped_invalid"],
            failed_urls=totals["failed_urls"],
        )

    @staticmethod
    def _accumulate(totals: dict[str, int], run_result: IngestionRunResult) -> None:
        totals["discovered_urls"] += run_result.discovered_urls
        totals["fetched_urls"] += run_result.fetched_urls
        totals["parsed_articles"] += run_result.parsed_articles
        totals["saved_articles"] += run_result.saved_articles
        totals["skipped_existing"] += run_result.skipped_existing
        totals["skipped_duplicates"] += run_result.skipped_duplicates
        totals["skipped_invalid"] += run_result.skipped_invalid
        totals["failed_urls"] += run_result.failed_urls
