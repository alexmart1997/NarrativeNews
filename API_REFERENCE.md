# API Reference - NarrativeNews

Справочник по основным API для работы с проектом.

---

## 📦 Инициализация

### Базовая инициализация

```python
from app.config.settings import get_settings
from app.db.connection import create_connection
from app.db.init_db import initialize_database
from app.bootstrap import build_app_services

# Получить конфиг из переменных окружения
settings = get_settings()

# Инициализировать БД (создать таблицы если нужно)
initialize_database(settings.database_path)

# Создать соединение и сервисы
with create_connection(settings.database_path) as conn:
    services = build_app_services(conn, settings)
    
    # Теперь можно использовать services.* для всех операций
```

---

## 📰 Работа с источниками и статьями

### Создание источника

```python
from app.models import SourceCreate

# Через репозиторий
source = services.source_repository.create(
    SourceCreate(
        name="Lenta.ru",
        domain="lenta.ru",
        base_url="https://lenta.ru",
        source_type="news_site",
        language="ru",
        is_active=True
    )
)

print(f"Источник создан с ID: {source.id}")
```

### Получение источников

```python
# По ID
source = services.source_repository.get_by_id(1)

# По домену
source = services.source_repository.get_by_domain("lenta.ru")

# Все источники
all_sources = services.source_repository.list()

# Только активные
active_sources = services.source_repository.list(only_active=True)
```

### Создание статьи

```python
from app.models import ArticleCreate
from app.services.normalization import ArticleNormalizer
from app.utils.text import estimate_word_count
import hashlib

# Подготовка данных
normalizer = ArticleNormalizer()

body_text = "Текст статьи..."
content_hash = hashlib.sha256(normalizer.normalize_text(body_text).encode()).hexdigest()

article = services.article_repository.create_article(
    ArticleCreate(
        source_id=1,  # ID источника
        url="https://lenta.ru/news/2026/04/29/example",
        title="Заголовок статьи",
        subtitle="Подзаголовок",
        body_text=body_text,
        published_at="2026-04-29T10:00:00",
        author="Редакция",
        category="политика",
        language="ru",
        content_hash=content_hash,
        word_count=estimate_word_count(body_text),
        is_canonical=True
    )
)

print(f"Статья создана с ID: {article.id}")
```

### Получение статей

```python
# По ID
article = services.article_repository.get_article_by_id(1)

# По URL
article = services.article_repository.get_article_by_url("https://...")

# По хешу контента (для дедупликации)
article = services.article_repository.get_article_by_content_hash("abc123...")

# За период (включая дубликаты)
articles = services.article_repository.list_articles_by_date_range(
    date_from="2026-04-20",
    date_to="2026-04-27"
)

# Только канонические статьи
canonical = services.article_repository.list_canonical_articles_by_date_range(
    date_from="2026-04-20",
    date_to="2026-04-27"
)

# Канонические статьи от конкретных источников
from_sources = services.article_repository.list_canonical_articles_by_date_range_and_sources(
    date_from="2026-04-20",
    date_to="2026-04-27",
    source_domains=["lenta.ru", "ria.ru"]
)
```

---

## ✂️ Разбиение статей на чанки

### Chunking Service

```python
from app.services.chunking import ChunkingService, ChunkingConfig

# Создать сервис с кастомной конфигурацией
chunking_service = ChunkingService(
    ChunkingConfig(
        target_chunk_size=700,  # Целевой размер в токенах
        min_chunk_size=300      # Минимальный размер
    )
)

# Разбить статью на чанки
chunks = chunking_service.chunk_article(article)

# Сохранить чанки в БД
saved_chunks = services.article_chunk_repository.create_many(chunks)

print(f"Создано {len(saved_chunks)} чанков для статьи {article.id}")
```

### Получение чанков

```python
# Все чанки одной статьи
chunks = services.article_chunk_repository.list_by_article_id(article_id)

# Чанки без эмбеддингов (для батч-генерации)
chunks_to_embed = services.article_chunk_repository.list_chunks_without_embeddings(
    model_name="nomic-embed-text",
    limit=100
)
```

---

## 🔍 RAG поиск

### Simple Search

```python
# Поиск чанков
results = services.rag_service.search_chunks(
    query="экономический кризис",
    date_from="2026-04-20",
    date_to="2026-04-27",
    limit=10,
    source_domains=["lenta.ru", "ria.ru"]  # опционально
)

# Каждый результат содержит:
for chunk_result in results:
    print(f"Релевантность: {chunk_result.relevance_score:.2f}")
    print(f"Статья: {chunk_result.article_title}")
    print(f"Текст: {chunk_result.chunk_text}\n")
```

### Search with Articles

```python
# Поиск с возвратом полных статей
search_result = services.rag_service.search(
    query="экономический кризис",
    date_from="2026-04-20",
    date_to="2026-04-27",
    limit=5
)

# Релевантные чанки
for chunk in search_result.chunks:
    print(f"Чанк: {chunk.chunk_text[:100]}...")

# Полные статьи
for article in search_result.articles:
    print(f"Статья: {article.title} ({article.published_at})")
```

### Answer Generation

```python
# Поиск + синтез ответа через LLM
answer_result = services.rag_service.answer(
    query="экономический кризис и его влияние на занятость",
    date_from="2026-04-20",
    date_to="2026-04-27",
    limit=10,
    include_debug_chunks=True  # включить найденные чанки в результат
)

# Синтезированный ответ
print("Ответ LLM:")
print(answer_result.summary_text)

# Источники
print("\nИсточники:")
for article in answer_result.source_articles:
    print(f"- {article.title} ({article.source_id})")

# Debug чанки (если включены)
if answer_result.top_chunks:
    print("\nИспользованные чанки:")
    for chunk in answer_result.top_chunks:
        print(f"  {chunk.chunk_text[:80]}...")
```

---

## 🧠 Анализ нарративов

### Базовый анализ

```python
from app.bootstrap import build_narrative_intelligence_services

# Получить сервисы анализа нарративов
pipeline = build_narrative_intelligence_services(conn, settings)

# Запустить анализ корпуса
result = pipeline.run(
    date_from="2026-04-20",
    date_to="2026-04-27",
    source_domains=["lenta.ru", "ria.ru"]  # опционально
)

# NarrativeIntelligenceRunResult содержит:
print(f"Документы: {len(result.documents)}")
print(f"Темы: {len(result.topics)}")
print(f"Фреймы: {len(result.frames)}")
print(f"Кластеры: {len(result.clusters)}")
print(f"Метки: {len(result.labels)}")
```

### Результаты анализа

```python
# Темы (Topics)
for topic in result.topics:
    print(f"Тема {topic.topic_id}: {topic.label}")
    print(f"  Ключевые слова: {', '.join(topic.keywords)}")
    print(f"  Статей: {len(topic.article_ids)}")
    print(f"  Уверенность: {topic.confidence:.2f}\n")

# Нарративные фреймы
for frame in result.frames:
    print(f"Фрейм {frame.frame_id}:")
    print(f"  Основное утверждение: {frame.main_claim}")
    print(f"  Акторы: {', '.join(frame.actors)}")
    print(f"  Причина: {frame.cause}")
    print(f"  Механизм: {frame.mechanism}")
    print(f"  Следствие: {frame.consequence}")
    print(f"  Тональность: {frame.valence}\n")

# Нарративные кластеры
for cluster in result.clusters:
    label = next((l for l in result.labels if l.cluster_id == cluster.cluster_id), None)
    if label:
        print(f"Нарратив: {label.title}")
        print(f"  Резюме: {label.summary}")
        print(f"  Основное утверждение: {label.canonical_claim}")
        print(f"  Фреймов: {len(cluster.frame_ids)}")
        print(f"  Статей: {len([a for a in result.assignments if a.cluster_id == cluster.cluster_id and a.assigned])}\n")
```

### Динамика нарративов

```python
# Временные ряды (как развивается нарратив)
for dynamics in result.dynamics:
    print(f"Нарратив {dynamics.cluster_id}:")
    print(f"  Всего статей: {dynamics.total_articles}")
    print(f"  Темп роста: {dynamics.growth_rate}")
    print(f"  Стабильность: {dynamics.stability_score}\n")
    
    # Точки по периодам
    for point in dynamics.points:
        print(f"  {point.period_start} — {point.period_end}:")
        print(f"    Статей: {point.article_count}")
        print(f"    Доля в корпусе: {point.share_of_corpus:.1%}")
        print(f"    Диверсификация источников: {point.source_diversity:.2f}")
```

### Сохранение результатов

```python
from app.bootstrap import build_narrative_materialization_service

# Сохранить анализ в БД
materialization_service = build_narrative_materialization_service(conn)

run = materialization_service.save_snapshot(
    source_domains_key="lenta.ru,ria.ru",  # для группировки
    date_from="2026-04-20",
    date_to="2026-04-27",
    result=result
)

print(f"Анализ сохранен с ID run: {run.id}")
print(f"Статистика:")
print(f"  Документы: {run.documents_count}")
print(f"  Темы: {run.topics_count}")
print(f"  Фреймы: {run.frames_count}")
print(f"  Кластеры: {run.clusters_count}")
```

---

## 🔗 Эмбеддинги и индексирование

### Генерация эмбеддингов

```python
from app.services.embeddings import EmbeddingIndexService

embedding_service = services.embedding_index_service

# Генерировать эмбеддинги для всех чанков без них
embedding_service.generate_embeddings_for_unindexed_chunks(
    batch_size=32,
    max_chunks=1000  # лимит для демонстрации
)

# Или вручную:
from app.services.llm import EmbeddingError

# Получить чанки
chunks = services.article_chunk_repository.list_chunks_without_embeddings(
    model_name="nomic-embed-text",
    limit=100
)

# Генерировать эмбеддинги
for chunk in chunks:
    try:
        embedding = services.embedding_client.embed_text(chunk.chunk_text)
        services.article_chunk_repository.upsert_chunk_embedding(
            chunk_id=chunk.id,
            model_name="nomic-embed-text",
            embedding=embedding
        )
    except EmbeddingError as e:
        print(f"Ошибка эмбеддинга для чанка {chunk.id}: {e}")
```

---

## 🛠️ Утилиты

### Работа с текстом

```python
from app.utils.text import estimate_word_count
from app.services.normalization import ArticleNormalizer

# Оценка количества слов
text = "Текст для анализа..."
word_count = estimate_word_count(text)

# Нормализация URL и текста
normalizer = ArticleNormalizer()
clean_url = normalizer.normalize_url("https://example.com/news?utm_source=tg&id=123")
clean_text = normalizer.normalize_text(text)
```

### Дедупликация

```python
from app.services.deduplication import DeduplicationService

dedup_service = DeduplicationService(
    article_repository=services.article_repository
)

# Найти дубликаты для новой статьи
duplicates = dedup_service.find_duplicates(new_article)

for dup in duplicates:
    print(f"Дубликат: {dup.duplicate_type} (оценка: {dup.similarity_score:.2f})")
```

---

## 🐍 Примеры для интеграции

### Flask API

```python
from flask import Flask, jsonify, request
from app.bootstrap import build_app_services
from app.db.connection import create_connection
from app.config.settings import get_settings

app = Flask(__name__)

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query')
    date_from = request.json.get('date_from')
    date_to = request.json.get('date_to')
    
    with create_connection(get_settings().database_path) as conn:
        services = build_app_services(conn, get_settings())
        
        results = services.rag_service.search_chunks(
            query=query,
            date_from=date_from,
            date_to=date_to,
            limit=10
        )
        
        return jsonify([{
            'text': r.chunk_text,
            'score': r.relevance_score,
            'article': r.article_title
        } for r in results])
```

### Batch Processing

```python
import json
from pathlib import Path

# Обработать статьи из JSON файла
def batch_ingest_articles(articles_json_path):
    settings = get_settings()
    initialize_database(settings.database_path)
    
    with open(articles_json_path) as f:
        articles_data = json.load(f)
    
    with create_connection(settings.database_path) as conn:
        services = build_app_services(conn, settings)
        
        for article_data in articles_data:
            article = services.article_repository.create_article(
                ArticleCreate(
                    source_id=article_data['source_id'],
                    url=article_data['url'],
                    title=article_data['title'],
                    subtitle=article_data.get('subtitle'),
                    body_text=article_data['body_text'],
                    published_at=article_data['published_at'],
                    # ...
                )
            )
            
            # Разбить на чанки
            chunks = services.chunking_service.chunk_article(article)
            services.article_chunk_repository.create_many(chunks)

batch_ingest_articles("articles.json")
```

---

## ⚠️ Обработка ошибок

```python
from app.services.llm import EmbeddingError, LLMError
from app.services.rag import RAGService

try:
    result = services.rag_service.answer(
        query="запрос",
        date_from="2026-04-20",
        date_to="2026-04-27"
    )
except (EmbeddingError, LLMError) as e:
    print(f"Ошибка при генерации: {e}")
    # Fallback на обычный поиск
    chunks = services.rag_service.search_chunks(...)
except RuntimeError as e:
    print(f"Ошибка БД: {e}")
```

---

## 📚 Дополнительная информация

- [README.md](README.md) - Быстрый старт
- [ARCHITECTURE.md](ARCHITECTURE.md) - Архитектура системы
- [examples/](examples/) - Полные рабочие примеры
