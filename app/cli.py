from __future__ import annotations

import argparse
from pathlib import Path

from app.config.logging import configure_logging
from app.config.settings import get_settings
from app.db.init_db import initialize_database
from app.db.connection import create_connection
from app.ingestion.fetcher import HttpFetcher
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.sources import SOURCE_CONFIGS, get_source_config
from app.repositories import ArticleChunkRepository, ArticleRepository, ClaimRepository, SourceRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NarrativeNews project utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_db_parser = subparsers.add_parser("init-db", help="Initialize the SQLite database.")
    init_db_parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Optional path to the SQLite database file.",
    )
    ingest_parser = subparsers.add_parser("ingest-source", help="Run one ingestion pass for a configured source.")
    ingest_parser.add_argument("source_name", choices=sorted(SOURCE_CONFIGS.keys()))
    ingest_parser.add_argument("--db-path", type=Path, default=None)
    ingest_parser.add_argument("--limit", type=int, default=10)

    return parser


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
            pipeline = IngestionPipeline(
                fetcher=HttpFetcher(),
                source_repository=SourceRepository(connection),
                article_repository=ArticleRepository(connection),
                article_chunk_repository=ArticleChunkRepository(connection),
                claim_repository=ClaimRepository(connection),
            )
            result = pipeline.run_once(get_source_config(args.source_name), limit=args.limit)
        print(result)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 1
