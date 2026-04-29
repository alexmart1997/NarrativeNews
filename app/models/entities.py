from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SourceCreate:
    """Модель для создания нового источника новостей.
    
    Атрибуты:
        name: Название источника (например, 'Lenta.ru')
        domain: Доменное имя (например, 'lenta.ru')
        base_url: Базовый URL сайта (например, 'https://lenta.ru')
        source_type: Тип источника (например, 'news_site', 'agency')
        language: Язык контента (по умолчанию 'ru')
        is_active: Активен ли источник для ингестии
    """
    name: str
    domain: str
    base_url: str
    source_type: str
    language: str = "ru"
    is_active: bool = True


@dataclass(slots=True)
class Source(SourceCreate):
    """Модель источника новостей с metadata из БД.
    
    Расширяет SourceCreate, добавляя:
        id: Уникальный ID в БД
        created_at: Дата создания записи
        updated_at: Дата последнего обновления
    """
    id: int = 0
    created_at: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class ArticleCreate:
    """Модель для создания новой новостной статьи.
    
    Атрибуты:
        source_id: ID источника из таблицы sources
        url: Уникальный URL статьи
        title: Заголовок статьи
        subtitle: Подзаголовок (может быть None)
        body_text: Основной текст статьи
        published_at: ISO-формат даты публикации
        author: Автор статьи (может быть None)
        category: Категория новости (может быть None)
        language: Язык контента (по умолчанию 'ru')
        content_hash: SHA256 хеш нормализованного контента для дедупликации
        word_count: Количество слов в статье
        is_canonical: Является ли это канонической версией статьи
        duplicate_group_id: ID группы дубликатов (если есть)
    """
    source_id: int
    url: str
    title: str
    subtitle: str | None
    body_text: str
    published_at: str
    author: str | None = None
    category: str | None = None
    language: str = "ru"
    content_hash: str = ""
    word_count: int = 0
    is_canonical: bool = True
    duplicate_group_id: str | None = None


@dataclass(slots=True)
class Article(ArticleCreate):
    """Модель статьи с metadata из БД.
    
    Расширяет ArticleCreate, добавляя:
        id: Уникальный ID в БД
        created_at: Дата добавления в БД
        updated_at: Дата последнего обновления
    """
    id: int = 0
    created_at: str = ""
    updated_at: str = ""
