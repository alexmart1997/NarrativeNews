from __future__ import annotations

import json
import re
from html import unescape
from typing import Any


COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b.*?>.*?</\1>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
JSON_LD_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(?P<body>.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)


def strip_tags(html_fragment: str) -> str:
    text = SCRIPT_STYLE_RE.sub(" ", html_fragment)
    text = COMMENT_RE.sub(" ", text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|section|li|h\d|blockquote)>", "\n", text, flags=re.IGNORECASE)
    text = TAG_RE.sub(" ", text)
    text = unescape(text)
    text = WHITESPACE_RE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(unescape(value).replace("\xa0", " ").split()).strip()


def extract_first_match(html: str, patterns: list[re.Pattern[str]]) -> str | None:
    for pattern in patterns:
        match = pattern.search(html)
        if match:
            for group_name in ("content", "value", "text"):
                try:
                    value = match.group(group_name)
                    if value:
                        return normalize_text(strip_tags(value))
                except IndexError:
                    continue
            value = match.group(1)
            if value:
                return normalize_text(strip_tags(value))
    return None


def extract_meta_content(html: str, names: list[str]) -> str | None:
    for name in names:
        pattern = re.compile(
            rf'<meta[^>]+(?:property|name)=["\']{re.escape(name)}["\'][^>]+content=["\'](?P<content>.*?)["\']',
            re.IGNORECASE | re.DOTALL,
        )
        reverse_pattern = re.compile(
            rf'<meta[^>]+content=["\'](?P<content>.*?)["\'][^>]+(?:property|name)=["\']{re.escape(name)}["\']',
            re.IGNORECASE | re.DOTALL,
        )
        for candidate in (pattern, reverse_pattern):
            match = candidate.search(html)
            if match:
                return normalize_text(match.group("content"))
    return None


def extract_json_ld_objects(html: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for match in JSON_LD_RE.finditer(html):
        raw = match.group("body").strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            objects.append(payload)
        elif isinstance(payload, list):
            objects.extend(item for item in payload if isinstance(item, dict))
    return objects


def extract_article_json_ld_value(html: str, key: str) -> str | None:
    for item in extract_json_ld_objects(html):
        item_type = item.get("@type")
        if item_type == "NewsArticle" or (isinstance(item_type, list) and "NewsArticle" in item_type):
            value = item.get(key)
            if isinstance(value, str):
                return normalize_text(value)
            if isinstance(value, dict):
                name = value.get("name")
                if isinstance(name, str):
                    return normalize_text(name)
    return None


def remove_blocks_by_pattern(html: str, patterns: list[re.Pattern[str]]) -> str:
    cleaned = html
    for pattern in patterns:
        cleaned = pattern.sub(" ", cleaned)
    return cleaned


def extract_attribute_value(tag_html: str, attribute_name: str) -> str | None:
    pattern = re.compile(
        rf'{re.escape(attribute_name)}=["\'](?P<value>.*?)["\']',
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(tag_html)
    if match:
        return normalize_text(match.group("value"))
    return None
