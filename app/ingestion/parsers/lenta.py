from __future__ import annotations

import re

from app.ingestion.parsers.base import BaseArticleParser, ParsedArticle
from app.ingestion.parsers.html_utils import (
    extract_article_json_ld_value,
    extract_attribute_value,
    extract_first_match,
    extract_meta_content,
    normalize_text,
    remove_blocks_by_pattern,
    strip_tags,
)


class LentaParser(BaseArticleParser):
    TITLE_PATTERNS = [
        re.compile(r'<h1[^>]+class="[^"]*topic-body__title[^"]*"[^>]*>(?P<content>.*?)</h1>', re.IGNORECASE | re.DOTALL),
        re.compile(r'<h1[^>]*>(?P<content>.*?)</h1>', re.IGNORECASE | re.DOTALL),
    ]
    SUBTITLE_PATTERNS = [
        re.compile(r'<div[^>]+class="[^"]*topic-body__title-topic[^"]*"[^>]*>(?P<content>.*?)</div>', re.IGNORECASE | re.DOTALL),
        re.compile(r'<p[^>]+class="[^"]*topic-body__subtitle[^"]*"[^>]*>(?P<content>.*?)</p>', re.IGNORECASE | re.DOTALL),
    ]
    AUTHOR_PATTERNS = [
        re.compile(r'<span[^>]+class="[^"]*topic-authors__name[^"]*"[^>]*>(?P<content>.*?)</span>', re.IGNORECASE | re.DOTALL),
        re.compile(r'<a[^>]+class="[^"]*topic-authors__name[^"]*"[^>]*>(?P<content>.*?)</a>', re.IGNORECASE | re.DOTALL),
    ]
    CATEGORY_PATTERNS = [
        re.compile(r'<a[^>]+class="[^"]*topic-header__rubric[^"]*"[^>]*>(?P<content>.*?)</a>', re.IGNORECASE | re.DOTALL),
        re.compile(r'<span[^>]+class="[^"]*topic-header__rubric[^"]*"[^>]*>(?P<content>.*?)</span>', re.IGNORECASE | re.DOTALL),
    ]
    BODY_CONTAINER_PATTERNS = [
        re.compile(r'<div[^>]+class="[^"]*topic-body__content[^"]*"[^>]*>(?P<content>.*?)</div>\s*</div>', re.IGNORECASE | re.DOTALL),
        re.compile(r'<section[^>]+class="[^"]*topic-page__content[^"]*"[^>]*>(?P<content>.*?)</section>', re.IGNORECASE | re.DOTALL),
    ]
    BODY_BLOCK_PATTERNS = [
        re.compile(r'<p[^>]*>(?P<content>.*?)</p>', re.IGNORECASE | re.DOTALL),
        re.compile(r'<div[^>]+class="[^"]*topic-body__content-text[^"]*"[^>]*>(?P<content>.*?)</div>', re.IGNORECASE | re.DOTALL),
    ]
    REMOVE_PATTERNS = [
        re.compile(r'<[^>]+class="[^"]*(?:topic-footer|footer|b-banner|adfox|ads|social|share|topic-tags)[^"]*"[^>]*>.*?</[^>]+>', re.IGNORECASE | re.DOTALL),
        re.compile(r'<[^>]+class="[^"]*(?:topic-body__rating|topic-body__comments|topic-body__rightcol|latest-news)[^"]*"[^>]*>.*?</[^>]+>', re.IGNORECASE | re.DOTALL),
        re.compile(r'<div[^>]*>[^<]*(?:Что думаешь\? Оцени!|Последние новости)[\s\S]*?</div>', re.IGNORECASE | re.DOTALL),
    ]

    def parse(self, html: str, url: str) -> ParsedArticle:
        cleaned_html = remove_blocks_by_pattern(html, self.REMOVE_PATTERNS)
        title = self._extract_title(cleaned_html)
        subtitle = self._extract_subtitle(cleaned_html)
        body_text = self._extract_body(cleaned_html)
        published_at = self._extract_published_at(cleaned_html)
        author = self._extract_author(cleaned_html)
        category = self._extract_category(cleaned_html)
        return ParsedArticle(
            title=title,
            subtitle=subtitle,
            body_text=body_text,
            published_at=published_at,
            author=author,
            category=category,
        )

    def _extract_title(self, html: str) -> str:
        return (
            extract_first_match(html, self.TITLE_PATTERNS)
            or extract_meta_content(html, ["og:title", "twitter:title"])
            or extract_article_json_ld_value(html, "headline")
            or ""
        )

    def _extract_subtitle(self, html: str) -> str | None:
        return (
            extract_first_match(html, self.SUBTITLE_PATTERNS)
            or extract_meta_content(html, ["description", "og:description"])
        )

    def _extract_author(self, html: str) -> str | None:
        return extract_first_match(html, self.AUTHOR_PATTERNS) or extract_article_json_ld_value(html, "author")

    def _extract_category(self, html: str) -> str | None:
        return (
            extract_first_match(html, self.CATEGORY_PATTERNS)
            or extract_meta_content(html, ["article:section"])
        )

    def _extract_published_at(self, html: str) -> str:
        datetime_tag = re.search(r"<time[^>]*datetime=[\"'](?P<value>.*?)[\"'][^>]*>", html, re.IGNORECASE | re.DOTALL)
        if datetime_tag:
            value = extract_attribute_value(datetime_tag.group(0), "datetime")
            if value:
                return value
        return (
            extract_meta_content(html, ["article:published_time", "og:published_time"])
            or extract_article_json_ld_value(html, "datePublished")
            or ""
        )

    def _extract_body(self, html: str) -> str:
        candidates = []
        for pattern in self.BODY_CONTAINER_PATTERNS:
            match = pattern.search(html)
            if match:
                candidates.append(match.group("content"))
        candidates.append(html)

        for candidate in candidates:
            paragraphs = self._extract_body_blocks(candidate)
            if paragraphs:
                return "\n\n".join(paragraphs)
        return ""

    def _extract_body_blocks(self, html: str) -> list[str]:
        paragraphs: list[str] = []
        for pattern in self.BODY_BLOCK_PATTERNS:
            for match in pattern.finditer(html):
                text = normalize_text(strip_tags(match.group("content")))
                if self._is_valid_paragraph(text):
                    paragraphs.append(text)
        return self._deduplicate_preserving_order(paragraphs)

    @staticmethod
    def _is_valid_paragraph(text: str) -> bool:
        invalid_phrases = ("Что думаешь? Оцени!", "Последние новости")
        return len(text) >= 20 and all(phrase not in text for phrase in invalid_phrases)

    @staticmethod
    def _deduplicate_preserving_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result
