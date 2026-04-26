from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import date
import json
from pathlib import Path

from app.bootstrap import build_narrative_intelligence_services, build_narrative_materialization_service
from app.config.logging import configure_logging
from app.config.settings import get_settings
from app.db.connection import create_connection
from app.db.init_db import initialize_database
from app.ingestion.discovery import ArchiveDiscoveryService
from app.ingestion.fetcher import HttpFetcher
from app.ingestion.pipeline import BulkIngestionService, IngestionPipeline
from app.ingestion.sources import SOURCE_CONFIGS, get_source_config
from app.repositories import ArticleChunkRepository, ArticleRepository, SourceRepository
from app.services import EmbeddingIndexService, build_source_domains_key, create_embedding_client


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="News RAG project utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_db_parser = subparsers.add_parser("init-db", help="Initialize the SQLite database.")
    init_db_parser.add_argument("--db-path", type=Path, default=None)

    ingest_parser = subparsers.add_parser("ingest-source", help="Run one ingestion pass for a configured source.")
    ingest_parser.add_argument("source_name", choices=sorted(SOURCE_CONFIGS.keys()))
    ingest_parser.add_argument("--db-path", type=Path, default=None)
    ingest_parser.add_argument("--limit", type=int, default=10)
    ingest_parser.add_argument("--fetch-timeout", type=float, default=20.0)
    ingest_parser.add_argument("--fetch-retries", type=int, default=2)
    ingest_parser.add_argument("--retry-delay", type=float, default=1.5)
    ingest_parser.add_argument("--skip-chunks", action="store_true", help="Skip chunk generation during ingestion.")
    ingest_parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation for new chunks during ingestion.",
    )

    backfill_parser = subparsers.add_parser(
        "backfill-source",
        help="Run archive-based bulk ingestion for a source over a date range.",
    )
    backfill_parser.add_argument("source_name", choices=sorted(SOURCE_CONFIGS.keys()))
    backfill_parser.add_argument("--db-path", type=Path, default=None)
    backfill_parser.add_argument("--date-from", type=parse_date, required=True)
    backfill_parser.add_argument("--date-to", type=parse_date, required=True)
    backfill_parser.add_argument("--per-day-limit", type=int, default=200)
    backfill_parser.add_argument("--fetch-timeout", type=float, default=30.0)
    backfill_parser.add_argument("--fetch-retries", type=int, default=4)
    backfill_parser.add_argument("--retry-delay", type=float, default=2.0)
    backfill_parser.add_argument("--skip-chunks", action="store_true", help="Skip chunk generation during ingestion.")
    backfill_parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation for new chunks during ingestion.",
    )

    embedding_parser = subparsers.add_parser(
        "index-embeddings",
        help="Generate embeddings for chunks that do not have them yet.",
    )
    embedding_parser.add_argument("--db-path", type=Path, default=None)
    embedding_parser.add_argument("--limit", type=int, default=200)

    narrative_parser = subparsers.add_parser(
        "analyze-narratives",
        help="Run narrative intelligence over the existing corpus without changing the database schema.",
    )
    narrative_parser.add_argument("--db-path", type=Path, default=None)
    narrative_parser.add_argument("--date-from", required=True)
    narrative_parser.add_argument("--date-to", required=True)
    narrative_parser.add_argument(
        "--source-domains",
        default=None,
        help="Comma-separated source domains, e.g. ria.ru,lenta.ru",
    )
    narrative_parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional JSON file path for the full narrative intelligence result.",
    )

    materialize_parser = subparsers.add_parser(
        "materialize-narratives",
        help="Run narrative intelligence locally and save the resulting snapshot into the database.",
    )
    materialize_parser.add_argument("--db-path", type=Path, default=None)
    materialize_parser.add_argument("--date-from", required=True)
    materialize_parser.add_argument("--date-to", required=True)
    materialize_parser.add_argument(
        "--source-domains",
        default=None,
        help="Comma-separated source domains, e.g. ria.ru,lenta.ru",
    )

    article_cache_parser = subparsers.add_parser(
        "precompute-narrative-articles",
        help="Precompute article-level narrative frames and embeddings once, then reuse them across date ranges.",
    )
    article_cache_parser.add_argument("--db-path", type=Path, default=None)
    article_cache_parser.add_argument("--date-from", required=True)
    article_cache_parser.add_argument("--date-to", required=True)
    article_cache_parser.add_argument(
        "--source-domains",
        default=None,
        help="Comma-separated source domains, e.g. ria.ru,lenta.ru",
    )
    article_cache_parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute article-level analyses even if they already exist in the cache.",
    )

    return parser


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    settings = get_settings(database_path=args.db_path)
    configure_logging(settings)

    if args.command == "init-db":
        initialize_database(settings.database_path)
        return 0

    if args.command == "ingest-source":
        initialize_database(settings.database_path)
        with create_connection(settings.database_path) as connection:
            chunk_repository = ArticleChunkRepository(connection)
            embedding_index_service = EmbeddingIndexService(
                article_chunk_repository=chunk_repository,
                embedding_client=create_embedding_client(settings),
            )
            pipeline = IngestionPipeline(
                fetcher=HttpFetcher(
                    timeout_seconds=args.fetch_timeout,
                    max_retries=args.fetch_retries,
                    retry_delay_seconds=args.retry_delay,
                ),
                source_repository=SourceRepository(connection),
                article_repository=ArticleRepository(connection),
                article_chunk_repository=chunk_repository,
                embedding_index_service=embedding_index_service,
                enable_chunking=not args.skip_chunks,
                enable_embeddings=not args.skip_embeddings,
            )
            result = pipeline.run_once(get_source_config(args.source_name), limit=args.limit)
        print(result)
        return 0

    if args.command == "backfill-source":
        initialize_database(settings.database_path)
        with create_connection(settings.database_path) as connection:
            fetcher = HttpFetcher(
                timeout_seconds=args.fetch_timeout,
                max_retries=args.fetch_retries,
                retry_delay_seconds=args.retry_delay,
            )
            chunk_repository = ArticleChunkRepository(connection)
            embedding_index_service = EmbeddingIndexService(
                article_chunk_repository=chunk_repository,
                embedding_client=create_embedding_client(settings),
            )
            pipeline = IngestionPipeline(
                fetcher=fetcher,
                source_repository=SourceRepository(connection),
                article_repository=ArticleRepository(connection),
                article_chunk_repository=chunk_repository,
                embedding_index_service=embedding_index_service,
                enable_chunking=not args.skip_chunks,
                enable_embeddings=not args.skip_embeddings,
            )
            service = BulkIngestionService(
                pipeline=pipeline,
                archive_discovery_service=ArchiveDiscoveryService(fetcher),
            )
            result = service.run_for_date_range(
                get_source_config(args.source_name),
                date_from=args.date_from,
                date_to=args.date_to,
                per_day_limit=args.per_day_limit,
            )
        print(result)
        return 0

    if args.command == "index-embeddings":
        initialize_database(settings.database_path)
        with create_connection(settings.database_path) as connection:
            service = EmbeddingIndexService(
                article_chunk_repository=ArticleChunkRepository(connection),
                embedding_client=create_embedding_client(settings),
            )
            indexed = service.index_missing_embeddings(limit=args.limit)
        print(f"Indexed embeddings for {indexed} chunks.")
        return 0

    if args.command == "analyze-narratives":
        initialize_database(settings.database_path)
        source_domains = _parse_source_domains(args.source_domains)
        with create_connection(settings.database_path) as connection:
            pipeline = build_narrative_intelligence_services(connection, settings)
            result = pipeline.run(
                date_from=args.date_from,
                date_to=args.date_to,
                source_domains=source_domains,
            )
        if args.output_path is not None:
            args.output_path.parent.mkdir(parents=True, exist_ok=True)
            args.output_path.write_text(
                json.dumps(_to_jsonable(result), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Narrative intelligence result saved to {args.output_path}")
        else:
            print(
                json.dumps(
                    {
                        "documents": len(result.documents),
                        "topics": len(result.topics),
                        "frames": len(result.frames),
                        "clusters": len(result.clusters),
                        "labels": len(result.labels),
                        "assignments": len(result.assignments),
                        "dynamics": len(result.dynamics),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return 0

    if args.command == "materialize-narratives":
        initialize_database(settings.database_path)
        source_domains = _parse_source_domains(args.source_domains)
        source_domains_key = build_source_domains_key(source_domains)
        with create_connection(settings.database_path) as connection:
            pipeline = build_narrative_intelligence_services(connection, settings)
            materialization_service = build_narrative_materialization_service(connection)
            result = pipeline.run(
                date_from=args.date_from,
                date_to=args.date_to,
                source_domains=source_domains,
            )
            run = materialization_service.save_snapshot(
                source_domains_key=source_domains_key,
                date_from=args.date_from,
                date_to=args.date_to,
                result=result,
            )
        print(
            json.dumps(
                {
                    "run_id": run.id,
                    "source_domains_key": run.source_domains_key,
                    "date_from": run.date_from,
                    "date_to": run.date_to,
                    "documents": run.documents_count,
                    "topics": run.topics_count,
                    "frames": run.frames_count,
                    "clusters": run.clusters_count,
                    "labels": run.labels_count,
                    "assignments": run.assignments_count,
                    "dynamics": run.dynamics_count,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "precompute-narrative-articles":
        initialize_database(settings.database_path)
        source_domains = _parse_source_domains(args.source_domains)
        with create_connection(settings.database_path) as connection:
            pipeline = build_narrative_intelligence_services(connection, settings)
            stats = pipeline.materialize_article_analyses(
                date_from=args.date_from,
                date_to=args.date_to,
                source_domains=source_domains,
                force=args.force,
            )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 1


def _parse_source_domains(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    domains = [item.strip() for item in raw_value.split(",") if item.strip()]
    return domains or None


def _to_jsonable(value):
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value
