from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass(slots=True)
class ParsedArticle:
    title: str
    subtitle: str | None
    body_text: str
    published_at: str
    author: str | None = None
    category: str | None = None


class BaseArticleParser(ABC):
    @abstractmethod
    def parse(self, html: str, url: str) -> ParsedArticle:
        raise NotImplementedError
