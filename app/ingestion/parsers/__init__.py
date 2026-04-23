from app.ingestion.parsers.base import BaseArticleParser, ParsedArticle
from app.ingestion.parsers.factory import get_article_parser
from app.ingestion.parsers.lenta import LentaParser
from app.ingestion.parsers.ria import RiaParser

__all__ = [
    "BaseArticleParser",
    "ParsedArticle",
    "get_article_parser",
    "LentaParser",
    "RiaParser",
]
