# Архитектура NarrativeNews

Документ с описанием архитектуры системы анализа нарративов в русскоязычных новостях.

---

## 📐 Общая архитектура

### Слои приложения

```
┌─────────────────────────────────────────────────┐
│ Presentation Layer (UI)                         │
│ - Streamlit Web Interface                       │
│ - CLI Commands                                  │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│ Service Layer (Business Logic)                  │
│ - RAG Service (поиск + синтез)                 │
│ - Narrative Intelligence (анализ нарративов)   │
│ - Chunking, Embeddings, Deduplication          │
│ - LLM & Embedding Integration                  │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│ Repository Layer (Data Access)                  │
│ - ArticleRepository                            │
│ - ArticleChunkRepository                       │
│ - SourceRepository                             │
│ - NarrativeAnalysisRepository                  │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│ Persistence Layer                               │
│ - SQLite3 Database                             │
│ - Full-Text Search Index (FTS)                 │
└─────────────────────────────────────────────────┘
```

---

## 🔄 Основные потоки данных

### 1. Ингестия новостей

```
Источник (RSS/веб)
    ↓
HTTP Fetcher (с retry)
    ↓
Source-specific Parser (Lenta, RIA, etc.)
    ↓
Normalization (URL cleanup, hash)
    ↓
Deduplication (по хешу и URL)
    ↓
ArticleRepository.create_article()
    ↓
SQLite (articles table)
```

**Ключевые компоненты:**
- [app/ingestion/fetcher.py](app/ingestion/fetcher.py) - HTTP запросы
- [app/ingestion/parsers/](app/ingestion/parsers/) - Специфичные для источника парсеры
- [app/services/normalization.py](app/services/normalization.py) - Нормализация URL и контента
- [app/services/deduplication.py](app/services/deduplication.py) - Выявление дубликатов

### 2. Подготовка данных для поиска (RAG)

```
Article (из БД)
    ↓
ChunkingService.chunk_article()
    ↓
ArticleChunkRepository.create_many()
    ↓
EmbeddingIndexService.generate_embeddings()
    ↓
LocalEmbeddingClient (nomic-embed-text)
    ↓
ArticleChunkRepository.upsert_chunk_embedding()
    ↓
SQLite (article_chunks + embeddings)
    ↓
FTS Index (для BM25 поиска)
```

**Ключевые компоненты:**
- [app/services/chunking.py](app/services/chunking.py) - Разбиение на чанки
- [app/services/embeddings.py](app/services/embeddings.py) - Генерация эмбеддингов
- [app/repositories/article_chunk_repository.py](app/repositories/article_chunk_repository.py) - Хранение

### 3. RAG Search Pipeline

```
User Query
    ↓
RAGService.search_chunks()
    ↓
┌─ _extract_query_terms() - parse query
│
├─ _hybrid_retrieve() - get candidates
│  ├─ search_chunks_lexical() - BM25
│  ├─ search_chunks_vector() - semantic (if embedding_client)
│  └─ merge results
│
├─ _rerank() - cross-encoder reranking (if reranker)
│
├─ _filter_topical_chunks() - remove noise
│
├─ _diversify_chunks() - max N per article
│
└─ Return top-K results

RAGAnswerResult
    ├─ Summary (LLM synthesis)
    ├─ Source articles
    └─ Top chunks
```

**Ключевые компоненты:**
- [app/services/rag.py](app/services/rag.py) - Основной сервис поиска
- [app/services/reranker.py](app/services/reranker.py) - Cross-encoder переранжирование
- [app/repositories/article_chunk_repository.py](app/repositories/article_chunk_repository.py) - Search methods

### 4. Анализ нарративов

```
CorpusArticlePreprocessor
    ├─ Подготовка документов
    └─ Загрузка эмбеддингов чанков
        ↓
TopicDiscoveryBackend (BERTopic)
    ├─ UMAP (dimensionality reduction)
    ├─ HDBSCAN (clustering)
    └─ Generate topic labels
        ↓
LLMNarrativeFrameExtractor (Qwen)
    ├─ Extract frames per article
    ├─ Actors, cause, consequence
    └─ Generate embeddings
        ↓
HybridNarrativeClassifier
    ├─ Cluster frames (HDBSCAN + embeddings)
    └─ Create narrative clusters
        ↓
LLMNarrativeLabeler (Qwen)
    ├─ Generate cluster labels
    ├─ Canonical claims
    └─ Counter-narratives
        ↓
RollingWindowNarrativeDynamicsAnalyzer
    ├─ Aggregate by time periods
    ├─ Calculate metrics (diversity, intensity)
    └─ Detect bursts
        ↓
NarrativeAnalysisRun (saved to DB)
```

**Ключевые компоненты:**
- [app/services/narrative_intelligence.py](app/services/narrative_intelligence.py) - Основной пайплайн
- [app/bootstrap.py](app/bootstrap.py) - Инициализация

---

## 📊 Модели данных

### Базовые сущности

```
Source (источник новостей)
├─ id, name, domain, base_url
├─ source_type, language, is_active
└─ created_at, updated_at

Article (новостная статья)
├─ id, source_id, url, title
├─ subtitle, body_text, published_at
├─ author, category, language
├─ content_hash (для дедупликации)
├─ is_canonical, duplicate_group_id
└─ created_at, updated_at

ArticleChunk (семантический фрагмент)
├─ id, article_id, chunk_index
├─ chunk_text, char_start, char_end
├─ token_count
└─ created_at

ArticleChunkEmbedding (векторное представление)
├─ chunk_id, model_name
├─ embedding_json (JSON array)
└─ dimension
```

### Нарративные модели

```
NarrativeFrame (фреймы статьи)
├─ frame_id, article_id, topic_id
├─ main_claim, actors, cause, mechanism
├─ consequence, future_expectation
├─ valence, implications
└─ confidence

NarrativeCluster (группа похожих фреймов)
├─ cluster_id, frame_ids
├─ centroid_frame_id
└─ noise (boolean)

NarrativeClusterLabel (интерпретация)
├─ cluster_id, title, summary
├─ canonical_claim, typical_formulations
├─ key_actors, causal_chain
├─ dominant_tone, counter_narrative
└─ representative_examples

NarrativeDynamicsSeries (динамика во времени)
├─ cluster_id, period_points
├─ growth_rate, stability_score
└─ total_articles

NarrativeAnalysisRun (снимок анализа)
├─ id, date_from, date_to
├─ source_domains_key
├─ counts (documents, topics, frames, clusters, etc.)
├─ payload_json (полные результаты)
└─ created_at
```

---

## 🔌 Интеграции с внешними компонентами

### Ollama + Local LLM

```
LLM Client (app/services/llm.py)
    ↓
HTTP Request to Ollama (localhost:11434)
    ↓
Qwen 2.5 3B (or custom model)
    ↓
Response (text generation, JSON parsing)
```

**Использование:**
- Извлечение нарративных фреймов
- Генерация меток для кластеров
- Синтез ответов в RAG
- Структурированная генерация (JSON)

### Эмбеддинги

```
EmbeddingClient (app/services/llm.py)
    ↓
nomic-embed-text (via Ollama)
    ↓
768-dimensional vectors
    ↓
Storage in SQLite JSON
    ↓
Used for:
    - Semantic search (RAG)
    - Frame clustering
    - Narrative similarity
```

### Переранжирование

```
CrossEncoderChunkReranker
    ↓
BAAI/bge-reranker-v2-m3
    ↓
Query + Chunk → Relevance Score
    ↓
Reranks top-K candidates
```

---

## 🗄️ Конфигурация

### Переменные окружения

```bash
# Основные пути
DATABASE_PATH=./data/narrative.db

# LLM (Ollama)
LLM_HOST=localhost
LLM_PORT=11434
LLM_MODEL=qwen2.5:3b
LLM_TEMPERATURE=0.7

# Эмбеддинги
EMBEDDING_MODEL=nomic-embed-text

# RAG параметры
RAG_HYBRID_LIMIT=24          # max candidates for hybrid retrieval
RAG_RERANK_LIMIT=8           # max candidates for reranking
RAG_TOP_K=5                  # final results returned

# Chunking
CHUNK_TARGET_SIZE=700        # target chunk size in tokens
CHUNK_MIN_SIZE=300           # minimum chunk size

# Topic Discovery
TOPIC_MIN_CLUSTER_SIZE=12    # HDBSCAN min cluster size

# Логирование
LOG_LEVEL=INFO
```

**Загрузка:** [app/config/settings.py](app/config/settings.py)

---

## 🧪 Тестирование

### Структура тестов

```
tests/
├─ test_repositories.py      # Unit tests для репозиториев
├─ test_rag_retrieval.py     # RAG pipeline tests
├─ test_narrative_intelligence.py  # Full narrative analysis
├─ test_parsers.py           # Source parsers
├─ test_schema.py            # Database schema
├─ test_chunking.py          # Chunking logic
└─ test_normalization.py     # Normalization functions
```

### Запуск тестов

```bash
# Все тесты
python -m unittest discover -s tests -v

# Конкретный модуль
python -m unittest tests.test_rag_retrieval -v

# Conкретный тест
python -m unittest tests.test_rag_retrieval.RAGRetrievalTest.test_search -v
```

---

## 🚀 Инициализация приложения

### Начальная установка

```python
from app.config.settings import get_settings
from app.db.init_db import initialize_database
from app.bootstrap import build_app_services
from app.db.connection import create_connection

# 1. Получить конфиг из env
settings = get_settings()

# 2. Инициализировать БД (создать таблицы)
initialize_database(settings.database_path)

# 3. Создать соединение
with create_connection(settings.database_path) as conn:
    # 4. Построить все сервисы
    services = build_app_services(conn, settings)
    
    # 5. Использовать сервисы
    chunks = services.rag_service.search_chunks(
        query="экономический кризис",
        date_from="2026-04-20",
        date_to="2026-04-27"
    )
```

---

## 🔐 Безопасность и оптимизация

### БД оптимизации

- **WAL режим**: Лучшая конкурентность
- **Foreign keys**: Ссылочная целостность
- **Индексы**: FTS для быстрого поиска, B-tree на date fields
- **Batch inserts**: Для производительности

### Дедупликация

- **Content hash**: SHA256 нормализованного текста
- **URL matching**: Точное совпадение URL
- **Canonical flag**: Выбор основной версии

### Фильтрация шума

- **STOPWORDS**: Фильтрация служебных слов
- **BROAD_TERMS**: Исключение обширных геогр. терминов
- **NOISE_PATTERNS**: Regex для удаления шаблонов шума
- **ACTION_TERMS**: Фильтрация политических действий

---

## 📚 Дополнительные ресурсы

- [README.md](README.md) - Быстрый старт и примеры использования
- Примеры: [examples/](examples/) - Полнофункциональные примеры

### Ключевые файлы для изучения

1. **Архитектура:**
   - [app/bootstrap.py](app/bootstrap.py) - Инициализация
   - [app/config/settings.py](app/config/settings.py) - Конфигурация

2. **RAG поиск:**
   - [app/services/rag.py](app/services/rag.py) - Основной сервис
   - [app/repositories/article_chunk_repository.py](app/repositories/article_chunk_repository.py) - Хранилище

3. **Нарративный анализ:**
   - [app/services/narrative_intelligence.py](app/services/narrative_intelligence.py) - Пайплайн
   - [app/models/narrative_intelligence.py](app/models/narrative_intelligence.py) - Модели

4. **Интеграции:**
   - [app/services/llm.py](app/services/llm.py) - LLM и embeddings
   - [app/services/reranker.py](app/services/reranker.py) - Переранжирование

---

## 📝 Примечания

- Все компоненты должны использоваться через `AppServices` контейнер
- Коннекция к БД управляется контекстным менеджером
- LLM вызовы должны обрабатывать исключения (timeout, connection errors)
- Эмбеддинги кешируются в БД для оптимизации
- Нарративный анализ может быть дорогостоящим - используйте батчинг
