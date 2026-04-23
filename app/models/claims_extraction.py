from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SentenceContext:
    sentence_text: str
    paragraph_index: int


@dataclass(frozen=True, slots=True)
class ClaimDraft:
    claim_text: str
    normalized_claim_text: str
    claim_type: str
    extraction_confidence: float | None = None
    classification_confidence: float | None = None
    source_sentence: str | None = None
    source_paragraph_index: int | None = None
