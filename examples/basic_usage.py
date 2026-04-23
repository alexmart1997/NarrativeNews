from __future__ import annotations

from app.config.settings import get_settings
from app.db.connection import create_connection
from app.db.init_db import initialize_database
from app.models import ArticleCreate, SourceCreate
from app.repositories import ArticleRepository, SourceRepository
from app.utils.text import estimate_word_count


def main() -> None:
    settings = get_settings()
    initialize_database(settings.database_path)

    with create_connection(settings.database_path) as connection:
        source_repo = SourceRepository(connection)
        article_repo = ArticleRepository(connection)

        source = source_repo.create(
            SourceCreate(
                name="РБК",
                domain="rbc.ru",
                base_url="https://www.rbc.ru",
                source_type="news_site",
            )
        )

        article = article_repo.create_article(
            ArticleCreate(
                source_id=source.id,
                url="https://www.rbc.ru/politics/2026/04/23/example",
                title="Пример новости",
                subtitle="Короткий подзаголовок",
                body_text="Это пример текста статьи для демонстрации инициализации проекта.",
                published_at="2026-04-23T09:00:00",
                author="Редакция",
                category="politics",
                content_hash="demo-hash-001",
                word_count=estimate_word_count(
                    "Это пример текста статьи для демонстрации инициализации проекта."
                ),
            )
        )

        loaded_article = article_repo.get_article_by_id(article.id)
        print(f"Source created: {source.id} -> {source.name}")
        print(f"Article created: {article.id} -> {article.title}")
        print(f"Loaded article: {loaded_article}")


if __name__ == "__main__":
    main()
