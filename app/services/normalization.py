from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.ingestion.parsers import ParsedArticle
from app.utils.text import estimate_word_count


@dataclass(frozen=True, slots=True)
class NormalizedArticle:
    url: str
    title: str
    subtitle: str | None
    body_text: str
    published_at: str
    author: str | None
    category: str | None
    word_count: int
    content_hash: str


class ArticleNormalizer:
    def normalize(self, parsed_article: ParsedArticle, *, url: str) -> NormalizedArticle:
        normalized_url = self.normalize_url(url)
        title = self._normalize_text_line(parsed_article.title)
        subtitle = self._normalize_optional_text(parsed_article.subtitle)
        body_text = self._normalize_body_text(parsed_article.body_text)
        published_at = parsed_article.published_at.strip()
        author = self._normalize_optional_text(parsed_article.author)
        category = self._normalize_optional_text(parsed_article.category)
        word_count = estimate_word_count(body_text)
        content_hash = hashlib.sha256(body_text.encode("utf-8")).hexdigest()
        return NormalizedArticle(
            url=normalized_url,
            title=title,
            subtitle=subtitle,
            body_text=body_text,
            published_at=published_at,
            author=author,
            category=category,
            word_count=word_count,
            content_hash=content_hash,
        )

    def normalize_url(self, url: str) -> str:
        split = urlsplit(url.strip())
        scheme = (split.scheme or "https").lower()
        netloc = split.netloc.lower()
        path = split.path or "/"
        if path != "/" and path.endswith("/"):
            path = path[:-1]
        query_items = [
            (key, value)
            for key, value in parse_qsl(split.query, keep_blank_values=False)
            if not key.lower().startswith("utm_")
        ]
        query = urlencode(query_items, doseq=True)
        return urlunsplit((scheme, netloc, path, query, ""))

    def _normalize_body_text(self, body_text: str) -> str:
        body_text = body_text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [self._normalize_text_line(line) for line in body_text.split("\n")]
        lines = [line for line in lines if line]
        return "\n\n".join(lines)

    def _normalize_optional_text(self, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = self._normalize_text_line(value)
        return normalized or None

    @staticmethod
    def _normalize_text_line(text: str) -> str:
        return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()
