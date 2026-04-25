from __future__ import annotations

import re
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


COMPACT_DATETIME_RE = re.compile(r"^\d{8}T\d{4}$")


def normalize_datetime_bound(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return stripped
    if COMPACT_DATETIME_RE.match(stripped):
        return stripped
    if "T" not in stripped:
        digits = re.sub(r"\D", "", stripped)
        if len(digits) == 8:
            return f"{digits}T0000"
        return stripped

    date_part, time_part = stripped.split("T", 1)
    date_digits = re.sub(r"\D", "", date_part)
    time_digits = re.sub(r"\D", "", time_part)
    if len(date_digits) != 8:
        return stripped
    if len(time_digits) < 4:
        time_digits = (time_digits + "0000")[:4]
    else:
        time_digits = time_digits[:4]
    return f"{date_digits}T{time_digits}"


def compact_datetime_sql(column_name: str) -> str:
    return (
        f"CASE "
        f"WHEN instr({column_name}, '-') > 0 "
        f"THEN replace(replace(substr({column_name}, 1, 16), '-', ''), ':', '') "
        f"ELSE {column_name} "
        f"END"
    )
