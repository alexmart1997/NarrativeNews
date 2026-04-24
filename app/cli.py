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
from app.services import EmbeddingIndexService, create_embedding_client


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
    ingest_parser.add_argument(
        "--skip-chunks",
        action="store_true",
        help="Skip chunk generation during ingestion.",
    )
    ingest_parser.add_argument(
        "--skip-claims",
        action="store_true",
        help="Skip claim extraction during ingestion.",
    )
    ingest_parser.add_argument(
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
            chunk_repository = ArticleChunkRepository(connection)
            embedding_index_service = EmbeddingIndexService(
                article_chunk_repository=chunk_repository,
                embedding_client=create_embedding_client(settings),
            )
            pipeline = IngestionPipeline(
                fetcher=HttpFetcher(),
                source_repository=SourceRepository(connection),
                article_repository=ArticleRepository(connection),
                article_chunk_repository=chunk_repository,
                claim_repository=ClaimRepository(connection),
                embedding_index_service=embedding_index_service,
                enable_chunking=not args.skip_chunks,
                enable_claim_extraction=not args.skip_claims,
                enable_embeddings=not args.skip_embeddings,
            )
            result = pipeline.run_once(get_source_config(args.source_name), limit=args.limit)
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

    parser.error(f"Unsupported command: {args.command}")
    return 1
