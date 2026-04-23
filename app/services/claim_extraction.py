from __future__ import annotations

import re
from abc import ABC, abstractmethod

from app.models import Article, ClaimCreate, ClaimDraft, SentenceContext


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
TECHNICAL_PREFIXES = (
    "поделиться",
    "читать ria.ru",
    "подписывайтесь",
    "последние новости",
    "что думаешь",
    "фото",
    "видео",
)
CLAIM_TYPES = {"predictive", "causal", "meta", "other"}


class BaseClaimLLMClient(ABC):
    @abstractmethod
    def extract_claims(self, article: Article, sentences: list[SentenceContext]) -> list[ClaimDraft]:
        raise NotImplementedError


class SimpleHeuristicClaimLLMClient(BaseClaimLLMClient):
    def extract_claims(self, article: Article, sentences: list[SentenceContext]) -> list[ClaimDraft]:
        drafts: list[ClaimDraft] = []
        for context in sentences:
            sentence = context.sentence_text.strip()
            if not self._looks_claimworthy(sentence):
                continue
            normalized = self._normalize_claim(sentence)
            if not normalized:
                continue
            drafts.append(
                ClaimDraft(
                    claim_text=sentence,
                    normalized_claim_text=normalized,
                    claim_type=self._classify_claim_type(sentence),
                    extraction_confidence=0.72,
                    classification_confidence=0.68,
                    source_sentence=sentence,
                    source_paragraph_index=context.paragraph_index,
                )
            )
            if len(drafts) >= 5:
                break
        if drafts:
            return drafts

        fallback_context = self._select_fallback_sentence(sentences)
        if fallback_context is None:
            return []
        normalized = self._normalize_claim(fallback_context.sentence_text)
        if not normalized:
            return []
        return [
            ClaimDraft(
                claim_text=fallback_context.sentence_text,
                normalized_claim_text=normalized,
                claim_type="other",
                extraction_confidence=0.55,
                classification_confidence=0.6,
                source_sentence=fallback_context.sentence_text,
                source_paragraph_index=fallback_context.paragraph_index,
            )
        ]

    def _looks_claimworthy(self, sentence: str) -> bool:
        lowered = sentence.lower()
        if len(sentence) < 30 or len(sentence) > 280:
            return False
        if any(lowered.startswith(prefix) for prefix in TECHNICAL_PREFIXES):
            return False
        if not any(keyword in lowered for keyword in self._interesting_keywords()):
            return False
        return True

    @staticmethod
    def _interesting_keywords() -> tuple[str, ...]:
        return (
            "заяв",
            "сообщ",
            "счит",
            "прогноз",
            "ожида",
            "привед",
            "вызва",
            "из-за",
            "поэтому",
            "может",
            "будет",
            "рост",
            "снижен",
        )

    def _classify_claim_type(self, sentence: str) -> str:
        lowered = sentence.lower()
        if any(token in lowered for token in ("будет", "ожида", "прогноз", "может")):
            return "predictive"
        if any(token in lowered for token in ("из-за", "привело", "вызвало", "поэтому", "в результате")):
            return "causal"
        if any(token in lowered for token in ("заяв", "сообщ", "по данным", "отметил", "подчеркнул")):
            return "meta"
        return "other"

    @staticmethod
    def _normalize_claim(sentence: str) -> str:
        normalized = " ".join(sentence.replace("\xa0", " ").split()).strip(" .")
        normalized = re.sub(r"^(по словам [^,]+,\s*)", "", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"^(по данным [^,]+,\s*)", "", normalized, flags=re.IGNORECASE)
        if len(normalized) > 160:
            normalized = normalized[:157].rstrip() + "..."
        return normalized

    @staticmethod
    def _select_fallback_sentence(sentences: list[SentenceContext]) -> SentenceContext | None:
        meaningful = [sentence for sentence in sentences if len(sentence.sentence_text) >= 60]
        if not meaningful:
            return None
        return sorted(meaningful, key=lambda item: len(item.sentence_text), reverse=True)[0]


class ClaimExtractor:
    def __init__(self, llm_client: BaseClaimLLMClient | None = None) -> None:
        self.llm_client = llm_client or SimpleHeuristicClaimLLMClient()

    def extract(self, article: Article) -> list[ClaimCreate]:
        if not article.is_canonical:
            return []

        sentences = self._split_article_into_sentences(article.body_text)
        drafts = self.llm_client.extract_claims(article, sentences)
        return self._postprocess_drafts(article.id, drafts)

    def _split_article_into_sentences(self, body_text: str) -> list[SentenceContext]:
        contexts: list[SentenceContext] = []
        paragraphs = [paragraph.strip() for paragraph in body_text.split("\n\n") if paragraph.strip()]
        for paragraph_index, paragraph in enumerate(paragraphs):
            for raw_sentence in SENTENCE_SPLIT_RE.split(paragraph):
                sentence = " ".join(raw_sentence.split()).strip()
                if self._is_candidate_sentence(sentence):
                    contexts.append(SentenceContext(sentence_text=sentence, paragraph_index=paragraph_index))
        return contexts

    def _postprocess_drafts(self, article_id: int, drafts: list[ClaimDraft]) -> list[ClaimCreate]:
        results: list[ClaimCreate] = []
        seen_normalized: set[str] = set()
        for draft in drafts:
            claim_text = " ".join(draft.claim_text.split()).strip()
            normalized = " ".join(draft.normalized_claim_text.split()).strip()
            if not claim_text or not normalized:
                continue
            if len(normalized) > len(claim_text):
                normalized = claim_text[:160].rstrip()
            if normalized.lower() in seen_normalized:
                continue
            if not self._is_claim_valid(claim_text, normalized, draft.claim_type):
                continue
            seen_normalized.add(normalized.lower())
            results.append(
                ClaimCreate(
                    article_id=article_id,
                    claim_text=claim_text,
                    normalized_claim_text=normalized,
                    claim_type=draft.claim_type if draft.claim_type in CLAIM_TYPES else "other",
                    extraction_confidence=draft.extraction_confidence,
                    classification_confidence=draft.classification_confidence,
                    source_sentence=draft.source_sentence or claim_text,
                    source_paragraph_index=draft.source_paragraph_index,
                )
            )
        return results

    @staticmethod
    def _is_candidate_sentence(sentence: str) -> bool:
        lowered = sentence.lower()
        if len(sentence) < 20:
            return False
        if any(lowered.startswith(prefix) for prefix in TECHNICAL_PREFIXES):
            return False
        return True

    @staticmethod
    def _is_claim_valid(claim_text: str, normalized: str, claim_type: str) -> bool:
        lowered = claim_text.lower()
        if len(claim_text) < 20:
            return False
        if any(token in lowered for token in ("copyright", "видео", "фото", "поделиться")):
            return False
        if claim_type not in CLAIM_TYPES:
            return False
        return True
