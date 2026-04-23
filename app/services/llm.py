from __future__ import annotations

from abc import ABC, abstractmethod

from app.models import ChunkSearchResult


class BaseLLMClient(ABC):
    @abstractmethod
    def generate_answer(self, query: str, chunks: list[ChunkSearchResult]) -> str:
        raise NotImplementedError


class SimpleExtractiveLLMClient(BaseLLMClient):
    def generate_answer(self, query: str, chunks: list[ChunkSearchResult]) -> str:
        if not chunks:
            return "По выбранному периоду релевантные фрагменты не найдены."

        paragraphs: list[str] = []
        for chunk in chunks[:2]:
            text = chunk.chunk_text.strip()
            if not text:
                continue
            paragraphs.append(text)

        if not paragraphs:
            return "По выбранному периоду релевантные фрагменты не найдены."

        return "\n\n".join(paragraphs)
