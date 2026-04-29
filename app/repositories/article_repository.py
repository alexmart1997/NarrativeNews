from __future__ import annotations

import sqlite3

from app.models.entities import Article, ArticleCreate
from app.repositories.base import BaseRepository, bool_to_int, compact_datetime_sql, normalize_datetime_bound


class ArticleRepository(BaseRepository):
    """Репозиторий для управления новостными статьями в БД.
    
    Предоставляет методы для:
    - Создания и поиска статей
    - Фильтрации по дате, источнику, содержимому
    - Управления дубликатами и каноническими версиями
    - Работы с группами дубликатов
    
    Примечания:
        - Все статьи хранят нормализованный content_hash для дедупликации
        - is_canonical флаг указывает на основную версию статьи
        - Статьи связаны с источниками через source_id
    """
    def create_article(self, payload: ArticleCreate) -> Article:
        """Создать новую статью в БД.
        
        Args:
            payload: Данные для создания статьи
            
        Returns:
            Созданная статья с ID из БД
            
        Raises:
            RuntimeError: Если статья не была загружена обратно из БД
        """
        cursor = self.connection.execute(
            """
            INSERT INTO articles (
                source_id, url, title, subtitle, body_text, published_at, author, category,
                language, content_hash, word_count, is_canonical, duplicate_group_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.source_id,
                payload.url,
                payload.title,
                payload.subtitle,
                payload.body_text,
                payload.published_at,
                payload.author,
                payload.category,
                payload.language,
                payload.content_hash,
                payload.word_count,
                bool_to_int(payload.is_canonical),
                payload.duplicate_group_id,
            ),
        )
        self.connection.commit()
        article = self.get_article_by_id(cursor.lastrowid)
        if article is None:
            raise RuntimeError("Created article could not be loaded back from the database.")
        return article

    def get_article_by_id(self, article_id: int) -> Article | None:
        """Получить статью по ID.
        
        Args:
            article_id: ID статьи
            
        Returns:
            Статья или None если не найдена
        """
        row = self._fetch_one("SELECT * FROM articles WHERE id = ?", (article_id,))
        return self._row_to_article(row) if row else None

    def get_article_by_url(self, url: str) -> Article | None:
        """Получить статью по URL.
        
        Args:
            url: Уникальный URL статьи
            
        Returns:
            Статья или None если не найдена
        """
        row = self._fetch_one("SELECT * FROM articles WHERE url = ?", (url,))
        return self._row_to_article(row) if row else None

    def get_article_by_content_hash(self, content_hash: str) -> Article | None:
        """Получить статью по хешу контента.
        
        Используется для обнаружения дубликатов. Если есть несколько версий,
        возвращает каноническую версию.
        
        Args:
            content_hash: SHA256 хеш нормализованного контента
            
        Returns:
            Статья или None если не найдена
        """
        row = self._fetch_one(
            """
            SELECT *
            FROM articles
            WHERE content_hash = ?
            ORDER BY is_canonical DESC, id ASC
            LIMIT 1
            """,
            (content_hash,),
        )
        return self._row_to_article(row) if row else None

    def list_by_ids(self, article_ids: list[int]) -> list[Article]:
        """Получить несколько статей по их ID.
        
        Args:
            article_ids: Список ID статей
            
        Returns:
            Список статей в порядке убывания даты публикации
        """
        if not article_ids:
            return []
        placeholders = ", ".join(["?"] * len(article_ids))
        rows = self._fetch_all(
            f"SELECT * FROM articles WHERE id IN ({placeholders}) ORDER BY published_at DESC, id DESC",
            tuple(article_ids),
        )
        return [self._row_to_article(row) for row in rows]

    def list_articles_by_date_range(self, date_from: str, date_to: str) -> list[Article]:
        """Получить статьи за период (включая дубликаты).
        
        Args:
            date_from: ISO-формат начальной даты
            date_to: ISO-формат конечной даты
            
        Returns:
            Статьи в порядке возрастания даты
        """
        date_from = normalize_datetime_bound(date_from) or date_from
        date_to = normalize_datetime_bound(date_to) or date_to
        rows = self._fetch_all(
            f"""
            SELECT *
            FROM articles
            WHERE {compact_datetime_sql("published_at")} BETWEEN ? AND ?
            ORDER BY {compact_datetime_sql("published_at")} ASC, id ASC
            """,
            (date_from, date_to),
        )
        return [self._row_to_article(row) for row in rows]

    def list_canonical_articles_by_date_range(self, date_from: str, date_to: str) -> list[Article]:
        """Получить только канонические статьи за период.
        
        Исключает известные дубликаты (non-canonical версии).
        
        Args:
            date_from: ISO-формат начальной даты
            date_to: ISO-формат конечной даты
            
        Returns:
            Канонические статьи в порядке возрастания даты
        """
        date_from = normalize_datetime_bound(date_from) or date_from
        date_to = normalize_datetime_bound(date_to) or date_to
        rows = self._fetch_all(
            f"""
            SELECT *
            FROM articles
            WHERE {compact_datetime_sql("published_at")} BETWEEN ? AND ? AND is_canonical = 1
            ORDER BY {compact_datetime_sql("published_at")} ASC, id ASC
            """,
            (date_from, date_to),
        )
        return [self._row_to_article(row) for row in rows]

    def list_canonical_articles_by_date_range_and_sources(
        self,
        date_from: str,
        date_to: str,
        source_domains: list[str] | None = None,
    ) -> list[Article]:
        """Получить канонические статьи за период и от конкретных источников.
        
        Args:
            date_from: ISO-формат начальной даты
            date_to: ISO-формат конечной даты
            source_domains: Список доменов источников (например, ['lenta.ru', 'ria.ru']).
                           Если None, возвращает все источники.
            
        Returns:
            Канонические статьи в порядке возрастания даты
        """
        date_from = normalize_datetime_bound(date_from) or date_from
        date_to = normalize_datetime_bound(date_to) or date_to
        clauses = [
            f"{compact_datetime_sql('published_at')} BETWEEN ? AND ?",
            "is_canonical = 1",
        ]
        params: list[object] = [date_from, date_to]
        if source_domains:
            placeholders = ", ".join("?" for _ in source_domains)
            clauses.append(
                f"source_id IN (SELECT id FROM sources WHERE domain IN ({placeholders}))"
            )
            params.extend(source_domains)
        rows = self._fetch_all(
            f"""
            SELECT *
            FROM articles
            WHERE {' AND '.join(clauses)}
            ORDER BY {compact_datetime_sql("published_at")} ASC, id ASC
            """,
            tuple(params),
        )
        return [self._row_to_article(row) for row in rows]

    def mark_article_canonical(
        self,
        article_id: int,
        is_canonical: bool,
        duplicate_group_id: str | None = None,
    ) -> bool:
        """Обновить статус канонической версии статьи.
        
        Используется для управления дубликатами и выбора основной версии.
        
        Args:
            article_id: ID статьи для обновления
            is_canonical: Является ли эта версия канонической
            duplicate_group_id: ID группы дубликатов (если есть)
            
        Returns:
            True если статья была обновлена, False если статьи не существует
        """
        cursor = self.connection.execute(
            """
            UPDATE articles
            SET is_canonical = ?, duplicate_group_id = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (bool_to_int(is_canonical), duplicate_group_id, article_id),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def list_by_source(self, source_id: int, limit: int = 100) -> list[Article]:
        """Получить статьи конкретного источника.
        
        Args:
            source_id: ID источника
            limit: Максимальное количество статей (по умолчанию 100)
            
        Returns:
            Список статей источника в порядке убывания даты
        """
            FROM articles
            WHERE source_id = ?
            ORDER BY published_at DESC, id DESC
            LIMIT ?
            """,
            (source_id, limit),
        )
        return [self._row_to_article(row) for row in rows]

    def create_duplicate_record(
        self,
        *,
        duplicate_group_id: str,
        article_id: int,
        duplicate_type: str,
        is_primary: bool,
        similarity_score: float | None = None,
    ) -> int:
        """Создать запись о дубликате статьи.
        
        Используется для отслеживания взаимосвязи дубликатов и выбора основной версии.
        
        Args:
            duplicate_group_id: ID группы дубликатов
            article_id: ID статьи в этой группе
            duplicate_type: Тип дубликата (например, 'hash_match', 'url_match')
            is_primary: Является ли это основной версией
            similarity_score: Оценка схожести (от 0 до 1), если применимо
            
        Returns:
            ID созданной записи в таблице article_duplicates
        """
        cursor = self.connection.execute(
            """
            INSERT INTO article_duplicates (
                duplicate_group_id, article_id, duplicate_type, is_primary, similarity_score
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                duplicate_group_id,
                article_id,
                duplicate_type,
                bool_to_int(is_primary),
                similarity_score,
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    @staticmethod
    def _row_to_article(row: sqlite3.Row) -> Article:
        """Преобразовать строку БД в объект Article.
        
        Внутренний метод для конвертации результатов SQL-запроса в dataclass.
        
        Args:
            row: Строка из результата запроса SELECT *
            
        Returns:
            Объект Article с заполненными полями
        """
        return Article(
            id=row["id"],
            source_id=row["source_id"],
            url=row["url"],
            title=row["title"],
            subtitle=row["subtitle"],
            body_text=row["body_text"],
            published_at=row["published_at"],
            author=row["author"],
            category=row["category"],
            language=row["language"],
            content_hash=row["content_hash"],
            word_count=row["word_count"],
            is_canonical=bool(row["is_canonical"]),
            duplicate_group_id=row["duplicate_group_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
