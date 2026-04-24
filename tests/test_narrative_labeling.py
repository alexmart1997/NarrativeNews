from __future__ import annotations

import unittest

from app.models import Article, Claim, GroupedClaimCluster
from app.services import BaseLLMClient, NarrativeLabelingService


class MockNarrativeLabelingLLMClient(BaseLLMClient):
    def __init__(self, payload: dict[str, str] | None = None, *, should_fail: bool = False) -> None:
        self.payload = payload
        self.should_fail = should_fail
        self.calls: list[dict[str, object]] = []

    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        raise AssertionError("generate_text should not be called in this test")

    def generate_json(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if self.should_fail:
            raise RuntimeError("LLM unavailable")
        return self.payload or {}


def build_cluster() -> GroupedClaimCluster:
    article = Article(
        id=1,
        source_id=1,
        url="https://example.ru/a1",
        title="Инфляция и рынок",
        subtitle=None,
        body_text="Инфляция вырастет к лету.",
        published_at="2026-04-20T10:00:00",
        content_hash="hash",
        word_count=4,
        is_canonical=True,
    )
    claims = [
        Claim(
            id=1,
            article_id=1,
            claim_text="Инфляция вырастет к лету.",
            normalized_claim_text="инфляция вырастет к лету",
            claim_type="predictive",
            source_sentence="Инфляция вырастет к лету.",
            source_paragraph_index=0,
        ),
        Claim(
            id=2,
            article_id=1,
            claim_text="Аналитики ждут рост цен летом.",
            normalized_claim_text="аналитики ждут рост цен летом",
            claim_type="predictive",
            source_sentence="Аналитики ждут рост цен летом.",
            source_paragraph_index=0,
        ),
    ]
    return GroupedClaimCluster(
        claim_type="predictive",
        representative_text="инфляция вырастет к лету",
        cluster_summary="инфляция вырастет к лету аналитики ждут рост цен летом",
        claims=claims,
        representative_claims=claims,
        articles=[article],
        cluster_score=0.8,
    )


class NarrativeLabelingTests(unittest.TestCase):
    def test_labeling_for_one_cluster(self) -> None:
        mock_llm = MockNarrativeLabelingLLMClient(
            {
                "title": "Рост инфляции летом",
                "formulation": "Инфляция, вероятно, вырастет к лету. Аналитики ожидают рост цен.",
                "explanation": "Кластер опирается на несколько согласованных predictive claims.",
            }
        )
        service = NarrativeLabelingService(llm_client=mock_llm)

        label = service.label_cluster(build_cluster())

        self.assertEqual(label.title, "Рост инфляции летом")
        self.assertEqual(len(mock_llm.calls), 1)
        self.assertIn("Narrative type: predictive", mock_llm.calls[0]["prompt"])
        self.assertIn("Cluster summary:", mock_llm.calls[0]["prompt"])

    def test_fallback_behavior_without_llm(self) -> None:
        service = NarrativeLabelingService()

        label = service.label_cluster(build_cluster())

        self.assertEqual(label.title, "инфляция вырастет к лету")
        self.assertIn("predictive", label.explanation)

    def test_fallback_behavior_when_llm_unavailable(self) -> None:
        service = NarrativeLabelingService(
            llm_client=MockNarrativeLabelingLLMClient(should_fail=True)
        )

        label = service.label_cluster(build_cluster())

        self.assertEqual(label.title, "инфляция вырастет к лету")
        self.assertIn("claims", label.explanation)


if __name__ == "__main__":
    unittest.main()
