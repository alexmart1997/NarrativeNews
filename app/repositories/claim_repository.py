from __future__ import annotations

import sqlite3

from app.models.entities import Claim, ClaimCreate
from app.repositories.base import BaseRepository


class ClaimRepository(BaseRepository):
    def create(self, payload: ClaimCreate) -> Claim:
        cursor = self.connection.execute(
            """
            INSERT INTO claims (
                article_id, claim_text, normalized_claim_text, claim_type,
                extraction_confidence, classification_confidence, source_sentence, source_paragraph_index
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.article_id,
                payload.claim_text,
                payload.normalized_claim_text,
                payload.claim_type,
                payload.extraction_confidence,
                payload.classification_confidence,
                payload.source_sentence,
                payload.source_paragraph_index,
            ),
        )
        self.connection.commit()
        claim = self.get_by_id(cursor.lastrowid)
        if claim is None:
            raise RuntimeError("Created claim could not be loaded back from the database.")
        return claim

    def create_many(self, payloads: list[ClaimCreate]) -> list[Claim]:
        created: list[Claim] = []
        for payload in payloads:
            created.append(self.create(payload))
        return created

    def get_by_id(self, claim_id: int) -> Claim | None:
        row = self._fetch_one("SELECT * FROM claims WHERE id = ?", (claim_id,))
        return self._row_to_claim(row) if row else None

    def list_by_article_id(self, article_id: int) -> list[Claim]:
        rows = self._fetch_all(
            "SELECT * FROM claims WHERE article_id = ? ORDER BY id ASC",
            (article_id,),
        )
        return [self._row_to_claim(row) for row in rows]

    def list_by_type(self, claim_type: str) -> list[Claim]:
        rows = self._fetch_all(
            "SELECT * FROM claims WHERE claim_type = ? ORDER BY id ASC",
            (claim_type,),
        )
        return [self._row_to_claim(row) for row in rows]

    def list_for_article_ids(
        self,
        article_ids: list[int],
        *,
        exclude_claim_type: str | None = None,
    ) -> list[Claim]:
        if not article_ids:
            return []
        placeholders = ", ".join(["?"] * len(article_ids))
        query = f"SELECT * FROM claims WHERE article_id IN ({placeholders})"
        params: list[object] = list(article_ids)
        if exclude_claim_type is not None:
            query += " AND claim_type != ?"
            params.append(exclude_claim_type)
        query += " ORDER BY id ASC"
        rows = self._fetch_all(query, tuple(params))
        return [self._row_to_claim(row) for row in rows]

    @staticmethod
    def _row_to_claim(row: sqlite3.Row) -> Claim:
        return Claim(
            id=row["id"],
            article_id=row["article_id"],
            claim_text=row["claim_text"],
            normalized_claim_text=row["normalized_claim_text"],
            claim_type=row["claim_type"],
            extraction_confidence=row["extraction_confidence"],
            classification_confidence=row["classification_confidence"],
            source_sentence=row["source_sentence"],
            source_paragraph_index=row["source_paragraph_index"],
            created_at=row["created_at"],
        )
