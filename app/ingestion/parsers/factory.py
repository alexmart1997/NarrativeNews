from __future__ import annotations

from app.ingestion.parsers.base import BaseArticleParser
from app.ingestion.parsers.lenta import LentaParser
from app.ingestion.parsers.ria import RiaParser


def get_article_parser(parser_type: str) -> BaseArticleParser:
    if parser_type == "lenta":
        return LentaParser()
    if parser_type == "ria":
        return RiaParser()
    raise ValueError(f"Unsupported parser_type: {parser_type}")
