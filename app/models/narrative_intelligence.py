from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ArticleAnalysisDocument:
    """Документ для анализа нарративов (статья с чанками).
    
    Представляет статью подготовленную для извлечения нарративных фреймов.
    
    Атрибуты:
        article_id: ID статьи в БД
        source_id: ID источника
        source_name: Название источника (например, 'Lenta.ru')
        source_domain: Доменное имя источника
        title: Заголовок статьи
        subtitle: Подзаголовок статьи
        body_text: Полный текст статьи
        published_at: Дата публикации в ISO-формате
        category: Категория/раздел новости
        chunk_texts: Кортеж текстов семантических чанков
    """
    article_id: int
    source_id: int
    source_name: str
    source_domain: str
    title: str
    subtitle: str | None
    body_text: str
    published_at: str
    category: str | None
    chunk_texts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TopicCandidate:
    """Кандидат на тему, обнаруженный в корпусе статей.
    
    Результат работы BERTopic + HDBSCAN для выявления основных смысловых линий.
    
    Атрибуты:
        topic_id: Уникальный ID темы
        label: Человекочитаемое название темы
        keywords: Ключевые слова, определяющие тему
        article_ids: ID статей, относящихся к этой теме
        confidence: Оценка уверенности в теме (0-1)
        metadata: Дополнительные метаданные (размер кластера, динамика и т.д.)
    """
    topic_id: str
    label: str
    keywords: tuple[str, ...]
    article_ids: tuple[int, ...]
    confidence: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NarrativeFrame:
    """Нарративный фрейм, извлеченный из статьи LLM.
    
    Представляет смысловую структуру и интерпретацию события/новости на уровне статьи.
    
    Атрибуты:
        frame_id: Уникальный ID фрейма
        article_id: ID статьи, из которой извлечен фрейм
        topic_id: ID темы/кластера (если назначен)
        status: Статус обработки ('extracted', 'validated', 'labeled')
        main_claim: Основное утверждение/позиция
        actors: Ключевые акторы/участники
        cause: Причина события
        mechanism: Механизм/процесс, которым происходит событие
        consequence: Последствия события
        future_expectation: Ожидаемое будущее развитие
        valence: Тональность ('positive', 'negative', 'neutral')
        implications: Подразумеваемые выводы
        representative_quotes: Цитаты из статьи, поддерживающие фрейм
        confidence: Уверенность LLM в качестве фрейма (0-1)
        metadata: Дополнительные метаданные
    """
    frame_id: str
    article_id: int
    topic_id: str | None
    status: str
    main_claim: str
    actors: tuple[str, ...]
    cause: str | None
    mechanism: str | None
    consequence: str | None
    future_expectation: str | None
    valence: str | None
    implications: tuple[str, ...]
    representative_quotes: tuple[str, ...] = ()
    confidence: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NarrativeFrameEmbedding:
    """Векторное представление нарративного фрейма.
    
    Позволяет проводить семантический поиск и кластеризацию на уровне нарративов.
    
    Атрибуты:
        frame_id: ID фрейма
        representation_text: Текстовое представление, использованное для эмбеддинга
        vector: Векторное представление в виде кортежа float'ов
        model_name: Название модели эмбеддингов (например, 'nomic-embed-text')
    """
    frame_id: str
    representation_text: str
    vector: tuple[float, ...]
    model_name: str


@dataclass(frozen=True, slots=True)
class NarrativeCluster:
    """Кластер схожих нарративных фреймов.
    
    Группирует фреймы с похожей смысловой структурой для анализа общих нарративных паттернов.
    
    Атрибуты:
        cluster_id: Уникальный ID кластера
        topic_id: Связанная тема (если есть)
        frame_ids: ID фреймов в этом кластере
        centroid_frame_id: ID наиболее репрезентативного фрейма
        noise: True если кластер содержит шум/выбросы
        metadata: Дополнительные метаданные (размер, плотность и т.д.)
    """
    cluster_id: str
    topic_id: str | None
    frame_ids: tuple[str, ...]
    centroid_frame_id: str | None
    noise: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NarrativeClusterLabel:
    """Интерпретивное описание нарративного кластера.
    
    Синтезированное человеческое описание, объясняющее, о чем говорят фреймы в кластере.
    
    Атрибуты:
        cluster_id: ID кластера
        title: Краткое название нарратива
        summary: Развернутое описание нарратива
        canonical_claim: Типичное основное утверждение
        typical_formulations: Типичные способы формулировки этого нарратива
        key_actors: Основные акторы в этом нарративе
        causal_chain: Цепь причинности (причина -> механизм -> следствие)
        dominant_tone: Доминирующая тональность
        counter_narrative: Противоположный нарратив (если существует)
        representative_examples: Примеры статей, хорошо представляющих этот нарратив
        metadata: Дополнительные метаданные
    """
    cluster_id: str
    title: str
    summary: str
    canonical_claim: str
    typical_formulations: tuple[str, ...]
    key_actors: tuple[str, ...]
    causal_chain: tuple[str, ...]
    dominant_tone: str | None
    counter_narrative: str | None
    representative_examples: tuple[str, ...]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NarrativeAssignment:
    """Назначение статьи к нарративному кластеру.
    
    Связывает статью с нарративом, в котором она участвует.
    
    Атрибуты:
        article_id: ID статьи
        frame_id: ID нарративного фрейма в этой статье
        cluster_id: ID кластера, к которому назначена статья
        similarity_score: Оценка схожести фрейма и кластера (0-1)
        assigned: True если эта статья активно участвует в нарративе
        reason: Причина назначения (если есть)
    """
    article_id: int
    frame_id: str
    cluster_id: str | None
    similarity_score: float
    assigned: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class NarrativeDynamicsPoint:
    """Точка данных для отслеживания динамики нарратива.
    
    Описывает, как нарратив развивается во времени в корпусе.
    
    Атрибуты:
        period_start: Начало периода (ISO-формат)
        period_end: Конец периода (ISO-формат)
        article_count: Количество статей с этим нарративом в период
        share_of_corpus: Доля в общем корпусе за период (0-1)
        source_diversity: Диверсификация по источникам (0-1)
        mean_intensity: Средняя уверенность фреймов в период
        burst_score: Оценка всплеска активности (если есть)
    """
    period_start: str
    period_end: str
    article_count: int
    share_of_corpus: float
    source_diversity: float
    mean_intensity: float | None = None
    burst_score: float | None = None


@dataclass(frozen=True, slots=True)
class NarrativeDynamicsSeries:
    """Временной ряд динамики одного нарративного кластера.
    
    Показывает, как изменяется активность нарратива с течением времени.
    
    Атрибуты:
        cluster_id: ID кластера, к которому относится этот ряд
        points: Точки данных за разные периоды времени
        total_articles: Всего статей с этим нарративом за весь период
        growth_rate: Скорость роста присутствия нарратива (если вычислена)
        stability_score: Оценка стабильности нарратива (0-1)
    """
    cluster_id: str
    points: tuple[NarrativeDynamicsPoint, ...]
    total_articles: int
    growth_rate: float | None = None
    stability_score: float | None = None


@dataclass(frozen=True, slots=True)
class NarrativeEvaluationReport:
    """Отчет о качестве результатов анализа нарративов.
    
    Содержит метрики для оценки надежности и полноты результатов.
    
    Атрибуты:
        topic_coherence: Связность обнаруженных тем (0-1)
        narrative_coherence: Связность нарративных кластеров (0-1)
        precision: Точность назначения статей к нарративам (0-1)
        recall: Полнота обнаружения нарративов (0-1)
        interpretability: Интерпретируемость результатов (0-1)
        novelty_detection_quality: Качество выявления новых нарративов (0-1)
        notes: Дополнительные примечания о результатах
    """
    topic_coherence: float | None = None
    narrative_coherence: float | None = None
    precision: float | None = None
    recall: float | None = None
    interpretability: float | None = None
    novelty_detection_quality: float | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NarrativeIntelligenceRunResult:
    """Полный результат анализа нарративов (снимок состояния).
    
    Содержит все артефакты, произведенные аналитическим пайплайном.
    
    Атрибуты:
        documents: Документы, использованные для анализа
        topics: Обнаруженные темы
        frames: Извлеченные нарративные фреймы
        embeddings: Векторные представления фреймов
        clusters: Кластеры схожих фреймов
        labels: Интерпретивные описания кластеров
        assignments: Назначения статей к кластерам
        dynamics: Временные ряды динамики нарративов
        evaluation: Оценка качества (если вычислена)
    """
    documents: tuple[ArticleAnalysisDocument, ...]
    topics: tuple[TopicCandidate, ...]
    frames: tuple[NarrativeFrame, ...]
    embeddings: tuple[NarrativeFrameEmbedding, ...]
    clusters: tuple[NarrativeCluster, ...]
    labels: tuple[NarrativeClusterLabel, ...]
    assignments: tuple[NarrativeAssignment, ...]
    dynamics: tuple[NarrativeDynamicsSeries, ...]
    evaluation: NarrativeEvaluationReport | None = None
