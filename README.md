# NarrativeNews

**Интеллектуальная система анализа нарративов в русскоязычных новостях**

Платформа для извлечения, анализа и отслеживания смысловых нарративов и фреймов в корпусе новостных статей. Система использует комбинацию методов машинного обучения (BERTopic, HDBSCAN, UMAP), локальные LLM (Ollama/Qwen), гибридный поиск и анализ динамики для выявления актуальных нарративных паттернов.

---

## 📦 Что включено в проект

### Архитектура и инфраструктура
- **SQLite3 база данных** с оптимизированной схемой (WAL-режим)
- **Модульная архитектура** с Repository Pattern
- **Dataclass-модели** с type hints для безопасности типов
- **Конфигурация через переменные окружения**

### Ингестия и парсинг
- **HTTP-фетчер** с retry-логикой и User-Agent спуфингом
- **RSS-дискавери** для автоматического обнаружения статей
- **Источник-специфичные парсеры** для Lenta.ru и RIA Novosti
- **Дедупликация** статей по URL и контент-хешу

### RAG (Retrieval Augmented Generation)
- **Гибридный поиск** (семантический + лексический)
- **Интеллектуальное разбиение статей** на семантические чанки
- **Cross-Encoder переранжирование** (BAAI BGE v2)
- **Локальные эмбеддинги** (nomic-embed-text)

### Анализ нарративов
- **Извлечение фреймов** через локальный LLM (Qwen 2.5 3B)
- **Обнаружение тем** (BERTopic + HDBSCAN + UMAP)
- **Кластеризация** нарративов
- **Анализ динамики** (частота, распределение по источникам, тренды)

### Интерфейсы
- **Streamlit MVP** для интерактивного анализа
- **CLI утилиты** для batch-обработки
- **REST-like JSON API** для интеграции

---

## 🚀 Быстрый старт

### Инициализация БД
```bash
python -m app init-db
```

### Пример работы с репозиториями
```bash
python -m examples.basic_usage
```

### Ингестия новостей
```bash
# Lenta.ru (первые 5 статей с архива)
python -m app ingest-source lenta --limit 5

# RIA Novosti (первые 5 статей)
python -m app ingest-source ria --limit 5
```

### Анализ нарративов
```bash
# Запустить анализ за период (по умолчанию: последние 7 дней)
python -m app materialize-narratives \
  --date-from 2026-04-20 \
  --date-to 2026-04-27

# С фильтром по источникам
python -m app materialize-narratives \
  --date-from 2026-04-20 \
  --date-to 2026-04-27 \
  --source-domains lenta.ru,ria.ru
```

### Запуск веб-интерфейса
```bash
streamlit run streamlit_app.py
```

### Тестирование
```bash
python -m unittest discover -s tests -v
```

---

## 🏗️ Архитектура системы

### Слои приложения

```
┌─────────────────────────────────────┐
│    UI Layer (Streamlit/CLI)         │
├─────────────────────────────────────┤
│    Services Layer                   │
│  - RAG, Narrative Intelligence      │
│  - Embeddings, Chunking, Dedup      │
├─────────────────────────────────────┤
│    Repository Layer                 │
│  - Article, Chunk, Analysis         │
├─────────────────────────────────────┤
│    Database (SQLite)                │
└─────────────────────────────────────┘
```

### Ключевые компоненты

| Компонент | Назначение | Технология |
|-----------|-----------|-----------|
| **ArticleRepository** | Управление статьями | SQLite |
| **ChunkingService** | Разбиение на чанки | Параграфный сплиттер |
| **RAGService** | Поиск и ранжирование | Vector + BM25 |
| **EmbeddingService** | Векторные представления | nomic-embed-text |
| **NarrativeIntelligence** | Извлечение нарративов | Qwen 2.5 LLM |
| **TopicDiscovery** | Обнаружение тем | BERTopic + HDBSCAN |

---

## ⚙️ Конфигурация

Параметры задаются через переменные окружения:

```bash
# Путь к БД
DATABASE_PATH=./data/narrative.db

# LLM (Ollama)
LLM_HOST=localhost
LLM_PORT=11434
LLM_MODEL=qwen2.5:3b

# Эмбеддинги
EMBEDDING_MODEL=nomic-embed-text

# RAG параметры
RAG_HYBRID_LIMIT=24
RAG_RERANK_LIMIT=8
RAG_TOP_K=5

# Chunking
CHUNK_TARGET_SIZE=700
CHUNK_MIN_SIZE=300

# Topic Discovery
TOPIC_MIN_CLUSTER_SIZE=12
```

---

## 📊 Структура проекта

```
NarrativeNews/
├── app/
│   ├── models/           # Dataclass модели (Article, NarrativeFrame, etc)
│   ├── repositories/     # Data access layer
│   ├── services/         # Бизнес-логика (RAG, Narrative Intelligence)
│   ├── ingestion/        # Парсинг и ингестия новостей
│   ├── db/              # БД инициализация и схема
│   ├── config/          # Конфигурация и логирование
│   └── ui/              # Streamlit интерфейс
├── examples/            # Примеры использования
├── tests/               # Unit-тесты
└── pyproject.toml       # Poetry конфиг
```

---

## 🔌 Зависимости

### Основные
- **Python >= 3.11**
- **Streamlit** - Web UI
- **SQLite3** - База данных
- **PyTorch + Transformers** - Deep Learning
- **BERTopic, HDBSCAN, UMAP** - Topic modeling
- **Ollama** - Локальный LLM

### Установка
```bash
poetry install
```

---

## 🧪 Тестирование

```bash
# Все тесты
python -m unittest discover -s tests -v

# Конкретный тест
python -m unittest tests.test_rag_retrieval -v

# С покрытием (если установлен coverage)
coverage run -m unittest discover && coverage report
```

---

## 📝 Примеры использования

### Базовое использование репозиториев
```python
from app.config.settings import get_settings
from app.db.connection import create_connection
from app.db.init_db import initialize_database
from app.models import ArticleCreate, SourceCreate
from app.repositories import ArticleRepository, SourceRepository

settings = get_settings()
initialize_database(settings.database_path)

with create_connection(settings.database_path) as conn:
    source_repo = SourceRepository(conn)
    article_repo = ArticleRepository(conn)
    
    # Создать источник
    source = source_repo.create(SourceCreate(
        name="РБК",
        domain="rbc.ru",
        base_url="https://www.rbc.ru",
        source_type="news_site"
    ))
    
    # Создать статью
    article = article_repo.create_article(ArticleCreate(
        source_id=source.id,
        url="https://example.com/news",
        title="Заголовок",
        subtitle="Подзаголовок",
        body_text="Текст статьи...",
        published_at="2026-04-29T10:00:00"
    ))
```

### RAG поиск
```python
from app.bootstrap import build_app_services

services = build_app_services(connection, settings)
rag_service = services.rag_service

results = rag_service.search(
    query="экономический кризис",
    date_from="2026-04-20",
    date_to="2026-04-27",
    limit=5
)

for result in results.chunk_results:
    print(f"Релевантность: {result.relevance_score:.2f}")
    print(f"Текст: {result.chunk_text}")
```

---

## 📖 Дополнительная информация

### Как работает анализ нарративов
1. **Ингестия** — загрузка и парсинг статей
2. **Чанкинг** — разбиение на смысловые фрагменты
3. **Эмбеддинг** — векторизация через трансформеры
4. **Topic Discovery** — выявление основных тем через BERTopic
5. **Frame Extraction** — извлечение нарративных фреймов через LLM
6. **Clustering** — группировка фреймов в нарративные кластеры
7. **Dynamics Analysis** — отслеживание динамики и трендов

### Требования к Ollama
```bash
# Установить Ollama (https://ollama.ai)
# Скачать модели
ollama pull qwen2.5:3b
ollama pull nomic-embed-text

# Запустить сервер (по умолчанию слушает :11434)
ollama serve
```

---

## 📄 Лицензия

Проект разработан в рамках Проектного практикума УРФУ.

---

## 👥 Контакты

Проект разрабатывается в рамках Уральского федерального университета.
