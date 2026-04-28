from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

from app.config.settings import Settings
from app.models import ChunkSearchResult


class RerankerError(RuntimeError):
    """Raised when a model-based reranker cannot score candidates."""


class BaseChunkReranker(ABC):
    @abstractmethod
    def score(self, query: str, candidates: list[ChunkSearchResult]) -> list[float]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class CrossEncoderRerankerConfig:
    model_name: str
    max_length: int = 512
    batch_size: int = 8


class CrossEncoderChunkReranker(BaseChunkReranker):
    def __init__(self, config: CrossEncoderRerankerConfig) -> None:
        self.config = config
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RerankerError("Cross-encoder reranker requires transformers and torch.") from exc

        self._torch = torch
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(config.model_name)
        except Exception as exc:
            raise RerankerError(f"Could not load reranker model '{config.model_name}'.") from exc

        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def score(self, query: str, candidates: list[ChunkSearchResult]) -> list[float]:
        if not candidates:
            return []

        scores: list[float] = []
        for start in range(0, len(candidates), self.config.batch_size):
            batch = candidates[start : start + self.config.batch_size]
            pairs = [(query, f"{candidate.article_title}\n{candidate.chunk_text}") for candidate in batch]
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with self._torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                if logits.ndim == 2 and logits.shape[1] == 1:
                    values = logits[:, 0]
                elif logits.ndim == 2:
                    values = logits[:, -1]
                else:
                    values = logits
                batch_scores = [float(self._sigmoid(value.item())) for value in values]
                scores.extend(batch_scores)
        return scores

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)


def create_reranker(settings: Settings) -> BaseChunkReranker | None:
    backend = settings.rag_reranker_backend.strip().lower()
    if backend in {"", "none", "disabled"}:
        return None
    if backend == "cross_encoder":
        return CrossEncoderChunkReranker(
            CrossEncoderRerankerConfig(
                model_name=settings.rag_reranker_model_name,
                max_length=settings.rag_reranker_max_length,
                batch_size=settings.rag_reranker_batch_size,
            )
        )
    raise RerankerError(f"Unsupported reranker backend: {settings.rag_reranker_backend}")
