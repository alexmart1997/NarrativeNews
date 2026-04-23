from __future__ import annotations

import re
from collections import defaultdict
from datetime import UTC, datetime

from app.models import (
    Article,
    Claim,
    ClaimClusterCreate,
    ClaimClusterItemCreate,
    GroupedClaimCluster,
    NarrativeResultArticleCreate,
    NarrativeResultCreate,
    NarrativeRunCreate,
)
from app.repositories import (
    ArticleRepository,
    ClaimClusterRepository,
    ClaimRepository,
    NarrativeResultRepository,
    NarrativeRunRepository,
)
from app.services.narrative_labeling import NarrativeLabelingService


class NarrativeScorer:
    def score_cluster(self, *, claim_count: int, article_count: int) -> float:
        return min(1.0, round(0.18 * claim_count + 0.22 * article_count, 4))


class ClaimGrouper:
    def __init__(self, scorer: NarrativeScorer | None = None) -> None:
        self.scorer = scorer or NarrativeScorer()

    def group(self, claims: list[Claim], articles_by_id: dict[int, Article]) -> list[GroupedClaimCluster]:
        grouped: dict[tuple[str, str], list[Claim]] = defaultdict(list)
        labels: dict[tuple[str, str], str] = {}

        for claim in claims:
            label = claim.normalized_claim_text or claim.claim_text
            normalized_key = self._normalize_key(label)
            existing_key = self._find_matching_key(grouped.keys(), claim.claim_type, normalized_key)
            key = existing_key or (claim.claim_type, normalized_key)
            grouped[key].append(claim)
            if key not in labels or len(label) < len(labels[key]):
                labels[key] = label

        clusters: list[GroupedClaimCluster] = []
        for key, grouped_claims in grouped.items():
            claim_type, _normalized = key
            unique_article_ids: list[int] = []
            seen_articles: set[int] = set()
            for claim in grouped_claims:
                if claim.article_id in articles_by_id and claim.article_id not in seen_articles:
                    seen_articles.add(claim.article_id)
                    unique_article_ids.append(claim.article_id)
            articles = [articles_by_id[article_id] for article_id in unique_article_ids]
            score = self.scorer.score_cluster(
                claim_count=len(grouped_claims),
                article_count=len(articles),
            )
            clusters.append(
                GroupedClaimCluster(
                    claim_type=claim_type,
                    representative_text=labels[key],
                    claims=grouped_claims,
                    articles=articles,
                    cluster_score=score,
                )
            )

        return sorted(
            clusters,
            key=lambda cluster: (cluster.claim_type, -cluster.cluster_score, -len(cluster.claims)),
        )

    @staticmethod
    def _normalize_key(text: str) -> str:
        normalized = re.sub(r"[^\w\s]", " ", text.lower())
        return " ".join(normalized.split())

    def _find_matching_key(
        self,
        existing_keys: object,
        claim_type: str,
        normalized_key: str,
    ) -> tuple[str, str] | None:
        target_tokens = set(normalized_key.split())
        for existing_type, existing_key in existing_keys:
            if existing_type != claim_type:
                continue
            if existing_key == normalized_key:
                return existing_type, existing_key
            existing_tokens = set(existing_key.split())
            if not existing_tokens or not target_tokens:
                continue
            overlap = len(existing_tokens & target_tokens) / max(len(existing_tokens), len(target_tokens))
            if overlap >= 0.8:
                return existing_type, existing_key
        return None


class NarrativeRunService:
    def __init__(
        self,
        article_repository: ArticleRepository,
        claim_repository: ClaimRepository,
        narrative_run_repository: NarrativeRunRepository,
        claim_cluster_repository: ClaimClusterRepository,
        narrative_result_repository: NarrativeResultRepository,
        claim_grouper: ClaimGrouper | None = None,
        narrative_scorer: NarrativeScorer | None = None,
        narrative_labeling_service: NarrativeLabelingService | None = None,
    ) -> None:
        self.article_repository = article_repository
        self.claim_repository = claim_repository
        self.narrative_run_repository = narrative_run_repository
        self.claim_cluster_repository = claim_cluster_repository
        self.narrative_result_repository = narrative_result_repository
        self.narrative_scorer = narrative_scorer or NarrativeScorer()
        self.claim_grouper = claim_grouper or ClaimGrouper(self.narrative_scorer)
        self.narrative_labeling_service = narrative_labeling_service or NarrativeLabelingService()

    def run(self, topic_text: str, date_from: str, date_to: str) -> dict[str, object]:
        run = self.narrative_run_repository.create(
            NarrativeRunCreate(
                topic_text=topic_text,
                date_from=date_from,
                date_to=date_to,
                run_status="running",
            )
        )

        articles = self.article_repository.search_canonical_articles_by_topic_and_date_range(
            topic_text,
            date_from,
            date_to,
        )
        articles_by_id = {article.id: article for article in articles}
        claims = self._filter_claims_for_topic(topic_text, list(articles_by_id.values()))
        grouped_clusters = self.claim_grouper.group(claims, articles_by_id)

        persisted_clusters = self._persist_clusters(run.id, grouped_clusters)
        persisted_results = self._persist_results(run.id, persisted_clusters)

        self.narrative_run_repository.update_status(
            run.id,
            "completed",
            articles_selected_count=len(articles),
            claims_selected_count=len(claims),
            finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )

        return {
            "run": self.narrative_run_repository.get_by_id(run.id),
            "articles": articles,
            "claims": claims,
            "clusters": self.claim_cluster_repository.list_by_run_id(run.id),
            "results": self.narrative_result_repository.list_by_run_id(run.id),
            "persisted_results": persisted_results,
        }

    def _filter_claims_for_topic(self, topic_text: str, articles: list[Article]) -> list[Claim]:
        article_ids = [article.id for article in articles]
        articles_by_id = {article.id: article for article in articles}
        claims = self.claim_repository.list_for_article_ids(article_ids, exclude_claim_type="other")
        topic_tokens = [token.lower() for token in topic_text.split() if token.strip()]
        if not topic_tokens:
            return claims

        filtered: list[Claim] = []
        for claim in claims:
            claim_haystack = " ".join(
                value.lower()
                for value in (claim.normalized_claim_text, claim.claim_text, claim.source_sentence)
                if value
            )
            article = articles_by_id.get(claim.article_id)
            article_haystack = " ".join(
                value.lower()
                for value in (
                    article.title if article else None,
                    article.subtitle if article else None,
                    article.body_text if article else None,
                    article.category if article else None,
                )
                if value
            )
            if any(token in claim_haystack or token in article_haystack for token in topic_tokens):
                filtered.append(claim)
        return filtered

    def _persist_clusters(self, run_id: int, clusters: list[GroupedClaimCluster]) -> list[tuple[int, GroupedClaimCluster]]:
        persisted: list[tuple[int, GroupedClaimCluster]] = []
        for cluster in clusters:
            created = self.claim_cluster_repository.create(
                ClaimClusterCreate(
                    run_id=run_id,
                    claim_type=cluster.claim_type,
                    cluster_label=cluster.representative_text,
                    cluster_summary=cluster.representative_text,
                    cluster_score=cluster.cluster_score,
                    claim_count=len(cluster.claims),
                    article_count=len(cluster.articles),
                )
            )
            representative_claim_id = cluster.claims[0].id if cluster.claims else None
            self.claim_cluster_repository.create_items(
                [
                    ClaimClusterItemCreate(
                        cluster_id=created.id,
                        claim_id=claim.id,
                        membership_score=1.0,
                        is_representative=claim.id == representative_claim_id,
                    )
                    for claim in cluster.claims
                ]
            )
            persisted.append((created.id, cluster))
        return persisted

    def _persist_results(
        self,
        run_id: int,
        persisted_clusters: list[tuple[int, GroupedClaimCluster]],
    ) -> list[object]:
        top_per_type: dict[str, tuple[int, GroupedClaimCluster]] = {}
        for cluster_id, cluster in persisted_clusters:
            current = top_per_type.get(cluster.claim_type)
            if current is None or cluster.cluster_score > current[1].cluster_score:
                top_per_type[cluster.claim_type] = (cluster_id, cluster)

        created_results = []
        for narrative_type in ("predictive", "causal", "meta"):
            selected = top_per_type.get(narrative_type)
            if selected is None:
                continue
            _cluster_id, cluster = selected
            label = self.narrative_labeling_service.label_cluster(cluster)
            result = self.narrative_result_repository.create(
                NarrativeResultCreate(
                    run_id=run_id,
                    narrative_type=narrative_type,
                    title=label.title,
                    formulation=label.formulation,
                    explanation=label.explanation,
                    strength_score=cluster.cluster_score,
                )
            )
            support_articles = cluster.articles[:5]
            self.narrative_result_repository.create_result_articles(
                [
                    NarrativeResultArticleCreate(
                        narrative_result_id=result.id,
                        article_id=article.id,
                        rank=index + 1,
                        selection_reason=f"Article supports the {narrative_type} cluster.",
                    )
                    for index, article in enumerate(support_articles)
                ]
            )
            created_results.append(result)
        return created_results
