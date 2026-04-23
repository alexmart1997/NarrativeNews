from __future__ import annotations

import sqlite3
from typing import Any


def bool_to_int(value: bool) -> int:
    return int(value)


class RepositoryError(RuntimeError):
    """Raised when repository operations fail."""


class BaseRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def _fetch_one(self, query: str, params: tuple[Any, ...]) -> sqlite3.Row | None:
        cursor = self.connection.execute(query, params)
        return cursor.fetchone()

    def _fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        cursor = self.connection.execute(query, params)
        return list(cursor.fetchall())
