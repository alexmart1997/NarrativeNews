from __future__ import annotations

import logging
from pathlib import Path

from app.db.connection import create_connection
from app.db.schema import create_schema

logger = logging.getLogger(__name__)


def initialize_database(database_path: Path) -> None:
    with create_connection(database_path) as connection:
        create_schema(connection)
    logger.info("Database initialized at %s", database_path)
