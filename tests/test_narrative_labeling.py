from __future__ import annotations

import unittest

from app.models import Article, Claim, GroupedClaimCluster
from app.services import BaseNarrativeLabelingLLMClient, NarrativeLabel, NarrativeLabelingService


class MockNarrativeLabelingLLMClient(BaseNarrativeLabelingLLMClient):
    def __init__(self, label: NarrativeLabel | None = None, *, should_fail: bool = False) -> None:
        self.label = label
        self.should_fail = should_fail
        self.calls: list[dict[str, object]] = []

    def generate_narrative_label(
        self,
        *,
        narrative_type: str,
        cluster_summary: str,
        representative_claims: list[str],
        article_count: int,
        claim_count: int,
    ) -> NarrativeLabel:
        self.calls.append(
            {
                "narrative_type": narrative_type,
                "cluster_summary": cluster_summary,
                "representative_claims": representative_claims,
                "article_count": article_count,
                "claim_count": claim_count,
            }
        )
        if self.should_fail:
            raise RuntimeError("LLM unavailable")
        if self.label is None:
            return NarrativeLabel(title="", formulation="", explanation="")
        return self.label


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
        claims=claims,
        articles=[article],
        cluster_score=0.8,
    )


class NarrativeLabelingTests(unittest.TestCase):
    def test_labeling_for_one_cluster(self) -> None:
        mock_llm = MockNarrativeLabelingLLMClient(
            NarrativeLabel(
                title="Рост инфляции к лету",
                formulation="Инфляция, вероятно, усилится к лету.",
                explanation="Кластер объединяет несколько однотипных predictive claims.",
            )
        )
        service = NarrativeLabelingService(llm_client=mock_llm)

        label = service.label_cluster(build_cluster())

        self.assertEqual(label.title, "Рост инфляции к лету")
        self.assertEqual(len(mock_llm.calls), 1)
        self.assertEqual(mock_llm.calls[0]["narrative_type"], "predictive")
        self.assertEqual(mock_llm.calls[0]["article_count"], 1)
        self.assertEqual(mock_llm.calls[0]["claim_count"], 2)

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
