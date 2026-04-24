from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
import math
import re

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
from app.services.llm import BaseEmbeddingClient
from app.services.narrative_labeling import NarrativeLabelingService


TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{2,}")
TOPIC_STOPWORDS = {
    "что",
    "как",
    "где",
    "когда",
    "почему",
    "зачем",
    "про",
    "по",
    "это",
    "эта",
    "этот",
    "эти",
    "есть",
    "и",
    "или",
    "но",
    "в",
    "во",
    "на",
    "с",
    "со",
    "о",
    "об",
    "обо",
    "для",
    "ли",
    "же",
}
GENERIC_CLAIM_PATTERNS = (
    "ранее сообщалось",
    "по его словам",
    "по ее словам",
    "по словам",
    "как сообщалось",
    "стало известно",
    "отмечается",
    "уточняется",
    "сообщает",
)
LIGHT_NORMALIZATION_ENDINGS = (
    "иями",
    "ями",
    "ами",
    "ого",
    "ему",
    "ому",
    "ими",
    "его",
    "ов",
    "ев",
    "ом",
    "ем",
    "ам",
    "ям",
    "ах",
    "ях",
    "ий",
    "ой",
    "ей",
    "ию",
    "ия",
    "ие",
    "а",
    "я",
    "у",
    "ю",
    "ы",
    "и",
    "е",
    "о",
    "й",
)


class NarrativeScorer:
    def score_cluster(
        self,
        *,
        claim_count: int,
        article_count: int,
        representative_count: int,
        avg_extraction_confidence: float,
        avg_classification_confidence: float,
    ) -> float:
        confidence_score = (avg_extraction_confidence + avg_classification_confidence) / 2
        support_score = min(1.0, article_count / max(claim_count, 1))
        representative_score = min(1.0, representative_count / 3)
        raw_score = (
            0.28 * min(1.0, claim_count / 4)
            + 0.30 * min(1.0, article_count / 3)
            + 0.22 * confidence_score
            + 0.12 * support_score
            + 0.08 * representative_score
        )
        return round(min(1.0, raw_score), 4)


class ClaimGrouper:
    def __init__(
        self,
        scorer: NarrativeScorer | None = None,
        embedding_client: BaseEmbeddingClient | None = None,
    ) -> None:
        self.scorer = scorer or NarrativeScorer()
        self.embedding_client = embedding_client

    def group(self, claims: list[Claim], articles_by_id: dict[int, Article]) -> list[GroupedClaimCluster]:
        filtered_claims = [claim for claim in claims if self._is_claim_informative(claim)]
        if not filtered_claims:
            return []

        filtered_claims.sort(key=self._claim_quality, reverse=True)
        embeddings = self._embed_claims(filtered_claims)

        clusters_by_type: dict[str, list[dict[str, object]]] = defaultdict(list)
        for claim in filtered_claims:
            claim_key = self._normalize_key(claim.normalized_claim_text or claim.claim_text)
            claim_embedding = embeddings.get(claim.id)

            best_cluster: dict[str, object] | None = None
            best_score = 0.0
            for cluster in clusters_by_type[claim.claim_type]:
                lexical_score = self._lexical_similarity(claim_key, cluster["normalized_keys"])  # type: ignore[arg-type]
                semantic_score = self._semantic_similarity(claim_embedding, cluster.get("embedding"))
                combined_score = max(lexical_score, semantic_score * 0.92)
                if lexical_score >= 0.78 or semantic_score >= 0.90 or (lexical_score >= 0.58 and semantic_score >= 0.78):
                    if combined_score > best_score:
                        best_score = combined_score
                        best_cluster = cluster

            if best_cluster is None:
                clusters_by_type[claim.claim_type].append(
                    {
                        "claims": [claim],
                        "normalized_keys": [claim_key],
                        "embedding": claim_embedding,
                    }
                )
                continue

            best_cluster["claims"].append(claim)  # type: ignore[index]
            best_cluster["normalized_keys"].append(claim_key)  # type: ignore[index]
            if best_cluster.get("embedding") is None and claim_embedding is not None:
                best_cluster["embedding"] = claim_embedding

        grouped_clusters: list[GroupedClaimCluster] = []
        for claim_type, raw_clusters in clusters_by_type.items():
            for raw_cluster in raw_clusters:
                grouped_claims = list(raw_cluster["claims"])  # type: ignore[arg-type]
                grouped_claims.sort(key=self._claim_quality, reverse=True)
                representative_claims = grouped_claims[: min(4, len(grouped_claims))]

                article_scores: dict[int, float] = defaultdict(float)
                for claim in grouped_claims:
                    article_scores[claim.article_id] += self._claim_quality(claim)

                article_ids = sorted(
                    article_scores,
                    key=lambda article_id: (article_scores[article_id], articles_by_id[article_id].published_at if article_id in articles_by_id else ""),
                    reverse=True,
                )
                articles = [articles_by_id[article_id] for article_id in article_ids if article_id in articles_by_id]
                if not articles:
                    continue

                cluster_summary = self._build_cluster_summary(representative_claims)
                representative_text = representative_claims[0].normalized_claim_text or representative_claims[0].claim_text
                avg_extraction = self._average_confidence(grouped_claims, "extraction_confidence")
                avg_classification = self._average_confidence(grouped_claims, "classification_confidence")
                cluster_score = self.scorer.score_cluster(
                    claim_count=len(grouped_claims),
                    article_count=len(articles),
                    representative_count=len(representative_claims),
                    avg_extraction_confidence=avg_extraction,
                    avg_classification_confidence=avg_classification,
                )

                grouped_clusters.append(
                    GroupedClaimCluster(
                        claim_type=claim_type,
                        representative_text=representative_text,
                        cluster_summary=cluster_summary,
                        claims=grouped_claims,
                        representative_claims=representative_claims,
                        articles=articles,
                        cluster_score=cluster_score,
                    )
                )

        return sorted(
            grouped_clusters,
            key=lambda cluster: (cluster.claim_type, -cluster.cluster_score, -len(cluster.articles), -len(cluster.claims)),
        )

    def _embed_claims(self, claims: list[Claim]) -> dict[int, list[float]]:
        if self.embedding_client is None or not claims:
            return {}

        payloads = [
            (claim.id, claim.normalized_claim_text or claim.claim_text)
            for claim in claims
        ]
        try:
            vectors = self.embedding_client.embed_texts([text for _claim_id, text in payloads])
        except Exception:
            return {}

        embeddings: dict[int, list[float]] = {}
        for (claim_id, _text), vector in zip(payloads, vectors, strict=False):
            if vector:
                embeddings[claim_id] = vector
        return embeddings

    @staticmethod
    def _normalize_key(text: str) -> str:
        normalized = re.sub(r"[^\w\s]", " ", text.lower())
        normalized = " ".join(normalized.split())
        tokens = [ClaimGrouper._normalize_token(token) for token in normalized.split()]
        return " ".join(token for token in tokens if token)

    @staticmethod
    def _normalize_token(token: str) -> str:
        for ending in LIGHT_NORMALIZATION_ENDINGS:
            if token.endswith(ending) and len(token) - len(ending) >= 4:
                return token[: -len(ending)]
        return token

    def _lexical_similarity(self, claim_key: str, existing_keys: list[str]) -> float:
        target_tokens = set(claim_key.split())
        if not target_tokens:
            return 0.0

        best = 0.0
        for existing_key in existing_keys:
            existing_tokens = set(existing_key.split())
            if not existing_tokens:
                continue
            intersection = len(target_tokens & existing_tokens)
            union = len(target_tokens | existing_tokens)
            if union == 0:
                continue
            similarity = intersection / union
            if similarity > best:
                best = similarity
        return best

    @staticmethod
    def _semantic_similarity(left: list[float] | None, right: list[float] | None) -> float:
        if left is None or right is None or len(left) != len(right):
            return 0.0
        numerator = 0.0
        left_norm = 0.0
        right_norm = 0.0
        for left_value, right_value in zip(left, right, strict=False):
            numerator += left_value * right_value
            left_norm += left_value * left_value
            right_norm += right_value * right_value
        if left_norm <= 0 or right_norm <= 0:
            return 0.0
        return numerator / math.sqrt(left_norm * right_norm)

    def _build_cluster_summary(self, representative_claims: list[Claim]) -> str:
        if not representative_claims:
            return ""
        summary_parts: list[str] = []
        seen_keys: set[str] = set()
        for claim in representative_claims:
            text = (claim.normalized_claim_text or claim.claim_text).strip()
            key = self._normalize_key(text)
            if not text or key in seen_keys:
                continue
            seen_keys.add(key)
            summary_parts.append(text)
            if len(summary_parts) >= 2:
                break
        return " ".join(summary_parts) if len(summary_parts) > 1 else summary_parts[0]

    def _claim_quality(self, claim: Claim) -> float:
        text = (claim.normalized_claim_text or claim.claim_text or "").strip()
        extraction = claim.extraction_confidence or 0.55
        classification = claim.classification_confidence or 0.55
        length_bonus = 0.0
        if 40 <= len(text) <= 180:
            length_bonus = 0.15
        elif 25 <= len(text) <= 220:
            length_bonus = 0.08
        type_bonus = 0.08 if claim.claim_type in {"predictive", "causal"} else 0.04
        return extraction * 0.4 + classification * 0.35 + length_bonus + type_bonus

    def _is_claim_informative(self, claim: Claim) -> bool:
        text = " ".join((claim.normalized_claim_text or claim.claim_text or "").split()).lower()
        if len(text) < 20:
            return False
        if any(fragment in text for fragment in GENERIC_CLAIM_PATTERNS):
            return False
        unique_tokens = {token for token in TOKEN_RE.findall(text)}
        if len(unique_tokens) < 3:
            return False
        return True

    @staticmethod
    def _average_confidence(claims: list[Claim], field_name: str) -> float:
        values = [
            float(getattr(claim, field_name))
            for claim in claims
            if getattr(claim, field_name) is not None
        ]
        if not values:
            return 0.55
        return sum(values) / len(values)


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
        embedding_client: BaseEmbeddingClient | None = None,
    ) -> None:
        self.article_repository = article_repository
        self.claim_repository = claim_repository
        self.narrative_run_repository = narrative_run_repository
        self.claim_cluster_repository = claim_cluster_repository
        self.narrative_result_repository = narrative_result_repository
        self.embedding_client = embedding_client
        self.narrative_scorer = narrative_scorer or NarrativeScorer()
        self.claim_grouper = claim_grouper or ClaimGrouper(self.narrative_scorer, embedding_client=embedding_client)
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
        topic_terms = self._extract_topic_terms(topic_text)
        if not topic_terms:
            return claims

        topic_embedding = self._embed_topic(topic_text)
        anchor_term = topic_terms[0]
        filtered: list[Claim] = []
        for claim in claims:
            claim_text = " ".join(
                value
                for value in (claim.normalized_claim_text, claim.claim_text, claim.source_sentence)
                if value
            )
            article = articles_by_id.get(claim.article_id)
            article_text = " ".join(
                value
                for value in (
                    article.title if article else None,
                    article.subtitle if article else None,
                    article.category if article else None,
                )
                if value
            )
            claim_overlap = self._topic_overlap_score(topic_terms, claim_text)
            article_overlap = self._topic_overlap_score(topic_terms, article_text)
            claim_anchor_overlap = self._topic_overlap_score([anchor_term], claim_text)
            article_anchor_overlap = self._topic_overlap_score([anchor_term], article_text)
            if self._claim_matches_topic(
                topic_terms=topic_terms,
                claim_anchor_overlap=claim_anchor_overlap,
                article_anchor_overlap=article_anchor_overlap,
                claim_overlap=claim_overlap,
                article_overlap=article_overlap,
            ):
                filtered.append(claim)
                continue

            if topic_embedding is not None and self.embedding_client is not None:
                try:
                    claim_embedding = self.embedding_client.embed_text(claim_text)
                except Exception:
                    claim_embedding = []
                if (
                    claim_embedding
                    and (claim_anchor_overlap > 0 or article_anchor_overlap > 0)
                    and self._cosine_similarity(topic_embedding, claim_embedding) >= 0.88
                ):
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
                    cluster_summary=cluster.cluster_summary,
                    cluster_score=cluster.cluster_score,
                    claim_count=len(cluster.claims),
                    article_count=len(cluster.articles),
                )
            )
            representative_claim_ids = {claim.id for claim in cluster.representative_claims}
            self.claim_cluster_repository.create_items(
                [
                    ClaimClusterItemCreate(
                        cluster_id=created.id,
                        claim_id=claim.id,
                        membership_score=round(self.claim_grouper._claim_quality(claim), 4),
                        is_representative=claim.id in representative_claim_ids,
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
        topic_terms = self._extract_topic_terms(self.narrative_run_repository.get_by_id(run_id).topic_text) if self.narrative_run_repository.get_by_id(run_id) else []
        topic_embedding = self._embed_topic(self.narrative_run_repository.get_by_id(run_id).topic_text) if self.narrative_run_repository.get_by_id(run_id) else None
        for cluster_id, cluster in persisted_clusters:
            if topic_terms and not self._cluster_matches_topic(
                topic_terms=topic_terms,
                topic_embedding=topic_embedding,
                cluster=cluster,
            ):
                continue
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
            support_articles = self._select_support_articles(cluster, narrative_type)
            self.narrative_result_repository.create_result_articles(
                [
                    NarrativeResultArticleCreate(
                        narrative_result_id=result.id,
                        article_id=article.id,
                        rank=index + 1,
                        selection_reason=f"Article strongly supports the {narrative_type} narrative.",
                    )
                    for index, article in enumerate(support_articles)
                ]
            )
            created_results.append(result)
        return created_results

    def _select_support_articles(self, cluster: GroupedClaimCluster, narrative_type: str) -> list[Article]:
        article_scores: dict[int, float] = defaultdict(float)
        articles_by_id = {article.id: article for article in cluster.articles}
        representative_claim_ids = {claim.id for claim in cluster.representative_claims}

        for claim in cluster.claims:
            score = self.claim_grouper._claim_quality(claim)
            if claim.id in representative_claim_ids:
                score += 0.2
            article_scores[claim.article_id] += score

        ranked_article_ids = sorted(
            article_scores,
            key=lambda article_id: (
                article_scores[article_id],
                articles_by_id[article_id].published_at if article_id in articles_by_id else "",
            ),
            reverse=True,
        )
        return [articles_by_id[article_id] for article_id in ranked_article_ids if article_id in articles_by_id][:5]

    def _extract_topic_terms(self, topic_text: str) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()
        for match in TOKEN_RE.finditer(topic_text.lower()):
            token = match.group(0)
            if token in TOPIC_STOPWORDS:
                continue
            normalized = ClaimGrouper._normalize_token(token)
            if normalized in TOPIC_STOPWORDS or normalized in seen:
                continue
            seen.add(normalized)
            terms.append(normalized)
        return terms

    def _topic_overlap_score(self, topic_terms: list[str], text: str) -> float:
        haystack = {ClaimGrouper._normalize_token(token) for token in TOKEN_RE.findall(text.lower())}
        if not topic_terms:
            return 0.0
        matches = sum(1 for term in topic_terms if term in haystack)
        return matches / len(topic_terms)

    @staticmethod
    def _claim_matches_topic(
        *,
        topic_terms: list[str],
        claim_anchor_overlap: float,
        article_anchor_overlap: float,
        claim_overlap: float,
        article_overlap: float,
    ) -> bool:
        if not topic_terms:
            return True
        if len(topic_terms) == 1:
            return claim_overlap >= 1.0 or article_overlap >= 1.0
        if claim_anchor_overlap >= 1.0:
            return True
        if claim_overlap >= 0.5 and claim_anchor_overlap > 0:
            return True
        if claim_anchor_overlap > 0 and article_overlap >= 0.5:
            return True
        if article_anchor_overlap > 0 and claim_overlap >= 0.25:
            return True
        if article_anchor_overlap > 0 and article_overlap >= 0.67:
            return True
        return False

    def _cluster_matches_topic(
        self,
        *,
        topic_terms: list[str],
        topic_embedding: list[float] | None,
        cluster: GroupedClaimCluster,
    ) -> bool:
        representative_text = " ".join(
            [
                cluster.representative_text,
                cluster.cluster_summary,
                " ".join((claim.normalized_claim_text or claim.claim_text) for claim in cluster.representative_claims[:3]),
            ]
        )
        article_text = " ".join(
            value
            for article in cluster.articles[:3]
            for value in (article.title, article.subtitle, article.category)
            if value
        )
        anchor_term = topic_terms[0]
        claim_overlap = self._topic_overlap_score(topic_terms, representative_text)
        article_overlap = self._topic_overlap_score(topic_terms, article_text)
        claim_anchor_overlap = self._topic_overlap_score([anchor_term], representative_text)
        article_anchor_overlap = self._topic_overlap_score([anchor_term], article_text)
        if self._claim_matches_topic(
            topic_terms=topic_terms,
            claim_anchor_overlap=claim_anchor_overlap,
            article_anchor_overlap=article_anchor_overlap,
            claim_overlap=claim_overlap,
            article_overlap=article_overlap,
        ):
            return True
        if topic_embedding is None or self.embedding_client is None:
            return False
        try:
            cluster_embedding = self.embedding_client.embed_text(representative_text)
        except Exception:
            return False
        return bool(cluster_embedding) and (claim_anchor_overlap > 0 or article_anchor_overlap > 0) and self._cosine_similarity(topic_embedding, cluster_embedding) >= 0.86

    def _embed_topic(self, topic_text: str) -> list[float] | None:
        if self.embedding_client is None:
            return None
        try:
            vector = self.embedding_client.embed_text(topic_text)
        except Exception:
            return None
        return vector or None

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        numerator = 0.0
        left_norm = 0.0
        right_norm = 0.0
        for left_value, right_value in zip(left, right, strict=False):
            numerator += left_value * right_value
            left_norm += left_value * left_value
            right_norm += right_value * right_value
        if left_norm <= 0 or right_norm <= 0:
            return 0.0
        return numerator / math.sqrt(left_norm * right_norm)
