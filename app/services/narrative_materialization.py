from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json

from app.models import NarrativeAnalysisRun, NarrativeIntelligenceRunResult
from app.repositories import NarrativeAnalysisRepository


class NarrativeMaterializationService:
    def __init__(self, repository: NarrativeAnalysisRepository) -> None:
        self.repository = repository

    def save_snapshot(
        self,
        *,
        source_domains_key: str,
        date_from: str,
        date_to: str,
        result: NarrativeIntelligenceRunResult,
    ) -> NarrativeAnalysisRun:
        payload_json = json.dumps(_to_jsonable(result), ensure_ascii=False)
        return self.repository.save_run(
            source_domains_key=source_domains_key,
            date_from=date_from,
            date_to=date_to,
            payload_json=payload_json,
            status="completed",
            documents_count=len(result.documents),
            topics_count=len(result.topics),
            frames_count=len(result.frames),
            clusters_count=len(result.clusters),
            labels_count=len(result.labels),
            assignments_count=len(result.assignments),
            dynamics_count=len(result.dynamics),
        )


def build_source_domains_key(source_domains: list[str] | None) -> str:
    if not source_domains:
        return "*"
    return ",".join(sorted(dict.fromkeys(domain.strip() for domain in source_domains if domain.strip())))


def _to_jsonable(value):
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value
