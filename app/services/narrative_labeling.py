from __future__ import annotations

from app.models import GroupedClaimCluster
from app.services.llm import BaseLLMClient, NarrativeLabel


class NarrativeLabelingService:
    def __init__(self, llm_client: BaseLLMClient | None = None) -> None:
        self.llm_client = llm_client

    def label_cluster(self, cluster: GroupedClaimCluster) -> NarrativeLabel:
        if self.llm_client is None:
            return self._fallback_label(cluster)

        representative_claims = [
            (claim.normalized_claim_text or claim.claim_text).strip()
            for claim in cluster.representative_claims[:5]
        ]
        prompt = self._build_prompt(cluster, representative_claims)
        system_prompt = (
            "Ты размечаешь narrative result на русском языке. "
            "Верни JSON-объект вида "
            '{"title": "...", "formulation": "...", "explanation": "..."} . '
            "Title должен быть коротким, 3-8 слов. "
            "Formulation: 1-2 предложения, без лишней общности. "
            "Explanation: коротко объясни, почему это сильный нарратив, опираясь только на claims и число статей. "
            "Не придумывай смысл вне claims и не расширяй политические выводы шире данных."
        )
        try:
            payload = self.llm_client.generate_json(
                prompt,
                system_prompt=system_prompt,
                temperature=0.15,
                max_tokens=260,
            )
        except Exception:
            return self._fallback_label(cluster)

        label = NarrativeLabel(
            title=str(payload.get("title", "")).strip(),
            formulation=str(payload.get("formulation", "")).strip(),
            explanation=str(payload.get("explanation", "")).strip(),
        )
        if not label.title or not label.formulation or not label.explanation:
            return self._fallback_label(cluster)
        return label

    @staticmethod
    def _build_prompt(cluster: GroupedClaimCluster, representative_claims: list[str]) -> str:
        lines = [
            f"Narrative type: {cluster.claim_type}",
            f"Cluster summary: {cluster.cluster_summary}",
            f"Representative text: {cluster.representative_text}",
            f"Article count: {len(cluster.articles)}",
            f"Claim count: {len(cluster.claims)}",
            "",
            "Representative claims:",
        ]
        for index, claim in enumerate(representative_claims, start=1):
            lines.append(f"{index}. {claim}")
        lines.append("")
        lines.append("Верни только JSON с title, formulation и explanation.")
        return "\n".join(lines)

    @staticmethod
    def _fallback_label(cluster: GroupedClaimCluster) -> NarrativeLabel:
        formulation = cluster.cluster_summary.strip() or cluster.representative_text.strip()
        title_base = cluster.representative_text.strip() or formulation
        explanation = (
            f"Нарратив типа {cluster.claim_type} поддержан {len(cluster.claims)} claims "
            f"из {len(cluster.articles)} статей."
        )
        return NarrativeLabel(
            title=" ".join(title_base.split()[:8]).strip() or cluster.claim_type,
            formulation=formulation,
            explanation=explanation,
        )
