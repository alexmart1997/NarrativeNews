from __future__ import annotations

import sqlite3

from app.models.entities import Source, SourceCreate
from app.repositories.base import BaseRepository, bool_to_int


class SourceRepository(BaseRepository):
    """Репозиторий для управления источниками новостей.
    
    Предоставляет методы для:
    - Создания и поиска источников
    - Фильтрации по активности
    - Управления статусом источников
    
    Примечания:
        - Каждый источник имеет уникальный domain
        - is_active флаг контролирует, используется ли источник в ингестии
    """
    def create(self, payload: SourceCreate) -> Source:
        """Создать новый источник.
        
        Args:
            payload: Данные для создания источника
            
        Returns:
            Созданный источник с ID из БД
            
        Raises:
            RuntimeError: Если источник не был загружен обратно из БД
        """
        cursor = self.connection.execute(
            """
            INSERT INTO sources (name, domain, base_url, source_type, language, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                payload.name,
                payload.domain,
                payload.base_url,
                payload.source_type,
                payload.language,
                bool_to_int(payload.is_active),
            ),
        )
        self.connection.commit()
        source = self.get_by_id(cursor.lastrowid)
        if source is None:
            raise RuntimeError("Created source could not be loaded back from the database.")
        return source

    def get_by_id(self, source_id: int) -> Source | None:
        """Получить источник по ID.
        
        Args:
            source_id: ID источника
            
        Returns:
            Источник или None если не найден
        """
        row = self._fetch_one("SELECT * FROM sources WHERE id = ?", (source_id,))
        return self._row_to_source(row) if row else None

    def get_by_domain(self, domain: str) -> Source | None:
        """Получить источник по доменному имени.
        
        Args:
            domain: Доменное имя (например, 'lenta.ru')
            
        Returns:
            Источник или None если не найден
        """
        row = self._fetch_one("SELECT * FROM sources WHERE domain = ?", (domain,))
        return self._row_to_source(row) if row else None

    def list(self, only_active: bool | None = None) -> list[Source]:
        """Получить список источников.
        
        Args:
            only_active: Если True, вернет только активные источники.
                        Если False, вернет только неактивные.
                        Если None, вернет все источники.
            
        Returns:
            Список источников, отсортированный по названию
        """
        if only_active is None:
            rows = self._fetch_all("SELECT * FROM sources ORDER BY name ASC")
        else:
            rows = self._fetch_all(
                "SELECT * FROM sources WHERE is_active = ? ORDER BY name ASC",
                (bool_to_int(only_active),),
            )
        return [self._row_to_source(row) for row in rows]

    def set_active_status(self, source_id: int, is_active: bool) -> bool:
        """Обновить статус активности источника.
        
        Args:
            source_id: ID источника
            is_active: Новый статус активности
            
        Returns:
            True если источник был обновлен, False если не найден
        """
        cursor = self.connection.execute(
            """
            UPDATE sources
            SET is_active = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (bool_to_int(is_active), source_id),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_source(row: sqlite3.Row) -> Source:
        """Преобразовать строку БД в объект Source.
        
        Внутренний метод для конвертации результатов SQL-запроса в dataclass.
        
        Args:
            row: Строка из результата запроса SELECT *
            
        Returns:
            Объект Source с заполненными полями
        """
        return Source(
            id=row["id"],
            name=row["name"],
            domain=row["domain"],
            base_url=row["base_url"],
            source_type=row["source_type"],
            language=row["language"],
            is_active=bool(row["is_active"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
