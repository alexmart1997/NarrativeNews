from pathlib import Path


slides = [
    {
        "title": "News Intelligence",
        "eyebrow": "Презентация проекта",
        "kind": "hero",
        "html": """
            <div class="hero-mark">AI Product Presentation</div>
            <h1 class="hero-title">News Intelligence</h1>
            <p class="hero-subtitle">AI-платформа для смыслового поиска, аналитики и выявления нарративов в новостном потоке.</p>
            <div class="hero-chips">
              <span>Semantic Retrieval</span>
              <span>Narrative Intelligence</span>
              <span>Local-first stack</span>
              <span>RAG + Analytics</span>
            </div>
            <div class="team-grid">
              <div class="person"><strong>Абиев Марик</strong><span>Machine Learning Engineer</span></div>
              <div class="person"><strong>Барабошкина Кристина</strong><span>Frontend Engineer</span></div>
              <div class="person"><strong>Варфоломеев Константин</strong><span>Backend Engineer</span></div>
              <div class="person"><strong>Владынцев Сергей</strong><span>Backend Engineer</span></div>
              <div class="person"><strong>Мартыненко Алексей</strong><span>Machine Learning Engineer, Product Manager</span></div>
              <div class="person"><strong>Подгорнов Владислав</strong><span>Market Analysis &amp; Product Positioning</span></div>
            </div>
        """,
    },
    {
        "title": "Идея продукта",
        "eyebrow": "Product Vision",
        "html": """
            <p class="lead">Мы разрабатываем систему интеллектуального анализа новостей, которая превращает большой поток публикаций в удобный аналитический инструмент.</p>
            <div class="grid two">
              <div class="card"><h3>Что делает продукт</h3><ul><li>собирает и структурирует новостной корпус;</li><li>позволяет искать материалы не только по словам, но и по смыслу;</li><li>формирует краткие аналитические сводки по запросу;</li><li>выявляет не только темы, но и устойчивые нарративы.</li></ul></div>
              <div class="card accent"><h3>Краткая формулировка</h3><p>News Intelligence — это AI-система, которая помогает не просто читать новости, а понимать, какие сюжеты, трактовки и смысловые конструкции доминируют в информационном поле.</p></div>
            </div>
        """,
    },
    {
        "title": "Проблема, которую мы решаем",
        "eyebrow": "Problem",
        "html": """
            <div class="grid three">
              <div class="card"><h3>Избыточный объём информации</h3><p>Публикаций слишком много, и вручную анализировать весь поток долго и дорого.</p></div>
              <div class="card"><h3>Слабая точность традиционного поиска</h3><p>Обычный поиск по ключевым словам не учитывает смысл и плохо работает на перефразировках.</p></div>
              <div class="card"><h3>Отсутствие интерпретационного уровня</h3><p>Даже если статьи найдены, остаётся вопрос: какую общую картину они формируют и какая интерпретация событий повторяется в потоке.</p></div>
            </div>
            <div class="note-band"><span>аналитики тратят время на ручной разбор</span><span>важные сигналы теряются в шуме</span><span>интерпретации и нарративы приходится выявлять вручную</span><span>стоимость аналитики растёт вместе с объёмом данных</span></div>
        """,
    },
    {
        "title": "Для кого наш продукт",
        "eyebrow": "Audience",
        "html": """
            <div class="grid two">
              <div class="card"><h3>Целевая аудитория</h3><ul><li>аналитические центры;</li><li>OSINT- и research-команды;</li><li>PR и GR-подразделения;</li><li>риск- и репутационные команды;</li><li>корпоративная стратегия и business intelligence;</li><li>исследователи в сферах политики, экономики, технологий и медиа.</li></ul></div>
              <div class="card"><h3>Пользовательская ценность</h3><ul><li>быстрее находить нужный контекст;</li><li>быстрее готовить аналитические справки;</li><li>отслеживать информационные риски;</li><li>понимать, какие интерпретации событий становятся доминирующими.</li></ul></div>
            </div>
        """,
    },
    {
        "title": "Что именно делает продукт",
        "eyebrow": "Functions",
        "html": """
            <div class="card"><h3>Основные функции News Intelligence</h3><ul class="cols"><li>сбор и хранение новостного корпуса;</li><li>разбиение материалов на аналитически удобные фрагменты;</li><li>гибридный поиск по словам и по смыслу;</li><li>генерация краткой аналитической сводки по найденным материалам;</li><li>выделение тематических областей в корпусе;</li><li>извлечение narrative frames из статей;</li><li>кластеризация повторяющихся нарративов;</li><li>отслеживание динамики нарративов во времени.</li></ul></div>
            <blockquote>Мы работаем не только с вопросом «о чём эта новость?», но и с вопросом «какую интерпретационную историю она рассказывает?»</blockquote>
        """,
    },
    {
        "title": "Что такое нарратив и почему это важно",
        "eyebrow": "Narrative Intelligence",
        "html": """
            <div class="grid two">
              <div class="card"><h3>Topic</h3><p>Отвечает на вопрос: о чём текст.</p></div>
              <div class="card accent"><h3>Narrative</h3><p>Отвечает на вопросы: что происходит, почему это происходит, кто действует, к чему это ведёт и какой эмоциональный фон формируется.</p></div>
            </div>
            <div class="grid two">
              <div class="card"><h3>Примеры интерпретаций</h3><ul><li>кризис как результат внешнего давления;</li><li>кризис как следствие внутренних ошибок;</li><li>кризис как сигнал будущих угроз;</li><li>кризис как возможность для перестройки системы.</li></ul></div>
              <div class="card"><h3>Практическая польза</h3><ul><li>видеть повторяющиеся трактовки событий;</li><li>находить доминирующие и конкурирующие версии происходящего;</li><li>отслеживать изменение framing во времени;</li><li>анализировать не только события, но и логику их подачи.</li></ul></div>
            </div>
        """,
    },
    {
        "title": "Конкуренты и рыночный контекст",
        "eyebrow": "Market",
        "html": """
            <div class="grid two">
              <div class="card"><h3>Новостные агрегаторы</h3><p>Собирают и показывают публикации. Удобны для чтения потока, но почти не дают интеллектуальной аналитики.</p></div>
              <div class="card"><h3>Системы медиамониторинга</h3><p>Отслеживают упоминания брендов, персон, тем. Хорошо работают как алертинг и мониторинг, но слабо раскрывают смысловую структуру новостей.</p></div>
              <div class="card"><h3>Поисковые и BI-решения по текстам</h3><p>Позволяют искать документы и строить отчёты. Сильны в инфраструктуре и фильтрации, но часто не дают narrative-level аналитики.</p></div>
              <div class="card"><h3>LLM-обёртки над поиском</h3><p>Умеют строить сводки, но без хорошего retrieval и структуры корпуса часто дают нестабильное качество.</p></div>
            </div>
        """,
    },
    {
        "title": "Наше позиционирование",
        "eyebrow": "Positioning",
        "html": """
            <p class="lead">Мы находимся между классическим медиамониторингом и исследовательской AI-аналитикой.</p>
            <div class="grid two">
              <div class="card"><h3>Что объединяет продукт</h3><ul><li>структурированный новостной корпус;</li><li>гибридный retrieval;</li><li>LLM-сводки;</li><li>narrative intelligence.</li></ul></div>
              <div class="card accent"><h3>Наше отличие</h3><ul><li>ищем не только по словам, но и по смыслу;</li><li>строим ответ по найденным материалам, а не “галлюцинируем”;</li><li>выделяем устойчивые интерпретационные конструкции, а не только темы;</li><li>строим слой аналитики поверх корпуса.</li></ul></div>
            </div>
            <blockquote>Если агрегаторы отвечают на вопрос «что опубликовано?», а классический поиск — «где статьи по теме?», то наш продукт отвечает на вопрос «какой смысловой и нарративный контекст формируется вокруг темы?»</blockquote>
        """,
    },
    {
        "title": "Почему продукт экономически полезен",
        "eyebrow": "Economics",
        "html": """
            <div class="metrics four"><div><strong>↓</strong><span>снижение доли ручного чтения и фильтрации</span></div><div><strong>↑</strong><span>ускорение подготовки аналитических справок</span></div><div><strong>×</strong><span>повышение пропускной способности аналитика</span></div><div><strong>!</strong><span>раннее обнаружение новых сюжетов и рисков</span></div></div>
            <div class="grid two">
              <div class="card"><h3>Практический эффект</h3><p>Один аналитик с такой системой может обработать заметно больший объём новостей, чем при ручной работе или использовании только базового keyword search.</p></div>
              <div class="card"><h3>Что экономит продукт</h3><ul><li>время;</li><li>человеческий ресурс;</li><li>стоимость первичного анализа;</li><li>стоимость missed signals в информационном поле.</li></ul></div>
            </div>
        """,
    },
    {
        "title": "Общая архитектура системы",
        "eyebrow": "Architecture",
        "html": """
            <p class="lead">Сначала мы строим качественный корпус и retrieval-слой, затем поверх него добавляем интеллектуальную аналитику.</p>
            <div class="steps">
              <div class="step"><span>1</span><div><h3>Data Layer</h3><p>Сбор и нормализация новостей, хранение корпуса, разбиение статей на чанки.</p></div></div>
              <div class="step"><span>2</span><div><h3>Retrieval Layer</h3><p>Построение эмбеддингов, гибридный retrieval, reranking и генерация сводки.</p></div></div>
              <div class="step"><span>3</span><div><h3>Analytics Layer</h3><p>Topic discovery, narrative extraction, embeddings, clustering, labeling и dynamics.</p></div></div>
            </div>
        """,
    },
    {
        "title": "Источники данных и хранение",
        "eyebrow": "Data Layer",
        "html": """
            <div class="grid two">
              <div class="card"><h3>Источник данных</h3><p>На текущем этапе основным источником новостей выступают РИА Новости.</p><ul><li>стабильный и крупный корпус;</li><li>однородный формат материалов;</li><li>быстрый запуск ingestion-пайплайна;</li><li>основа для тестирования retrieval и narrative-аналитики.</li></ul></div>
              <div class="card accent"><h3>SQLite</h3><p>Для хранения на текущем этапе выбрана SQLite.</p><ul><li>хранит статьи, чанки, эмбеддинги и narrative snapshots;</li><li>не требует отдельного сервера БД;</li><li>удобна для локальной разработки и отладки;</li><li>даёт лучший баланс между простотой, скоростью разработки и надёжностью.</li></ul></div>
            </div>
        """,
    },
    {
        "title": "Подготовка корпуса",
        "eyebrow": "Preprocessing",
        "html": """
            <div class="grid three">
              <div class="card"><h3>Нормализация</h3><ul><li>заголовок</li><li>дата публикации</li><li>источник</li><li>URL</li><li>основной текст</li><li>служебные метаданные</li></ul></div>
              <div class="card"><h3>Очистка</h3><ul><li>устранение дубликатов</li><li>нормализация полей</li><li>проверка консистентности корпуса</li></ul></div>
              <div class="card"><h3>Почему это важно</h3><p>Если корпус изначально не нормализован, retrieval, embeddings, clustering и narratives будут работать нестабильно.</p></div>
            </div>
        """,
    },
    {
        "title": "Разбиение статей на чанки",
        "eyebrow": "Chunking",
        "html": """
            <div class="grid two">
              <div class="card"><h3>Почему не статья целиком</h3><ul><li>в одной статье могут быть несколько смысловых линий;</li><li>релевантный ответ часто содержится только в одном абзаце;</li><li>длинный документ хуже подходит для точного поиска.</li></ul></div>
              <div class="card accent"><h3>Что даёт chunk-based подход</h3><ul><li>повышает точность retrieval;</li><li>позволяет находить именно те фрагменты, которые отвечают на запрос;</li><li>улучшает качество RAG-сводок.</li></ul></div>
            </div>
        """,
    },
    {
        "title": "Embeddings и семантическое представление текста",
        "eyebrow": "Semantic Layer",
        "html": """
            <div class="grid two">
              <div class="card"><h3>Что происходит</h3><p>Для каждого текстового чанка строится embedding — векторное представление текста.</p><div class="chips"><span>nomic-embed-text</span><span>bge-m3</span><span>Ollama</span></div></div>
              <div class="card"><h3>Зачем нужны embeddings</h3><ul><li>поиск не только по совпадению слов;</li><li>поиск по смысловой близости;</li><li>обработка перефразировок и близких формулировок.</li></ul></div>
            </div>
            <blockquote>Например, система может сблизить: «инфляция в России», «рост потребительских цен», «ускорение подорожания товаров» — даже если слова не совпадают буквально.</blockquote>
        """,
    },
    {
        "title": "Гибридный retrieval",
        "eyebrow": "RAG Core",
        "html": """
            <div class="grid two">
              <div class="card"><h3>Лексический поиск</h3><ul><li>ищет по словам, словоформам и прямым совпадениям;</li><li>хорошо работает на точных сущностях, названиях и редких терминах.</li></ul></div>
              <div class="card"><h3>Семантический поиск</h3><ul><li>использует embeddings;</li><li>ищет по смысловой близости;</li><li>находит перефразировки и близкие по смыслу формулировки.</li></ul></div>
            </div>
            <div class="steps compact">
              <div class="step"><span>A</span><div><p>Оба поиска формируют свои списки кандидатов по chunk-уровню.</p></div></div>
              <div class="step"><span>B</span><div><p>Кандидаты объединяются, и для каждого считается итоговая релевантность.</p></div></div>
              <div class="step"><span>C</span><div><p>Hybrid retrieval даёт лучший баланс точности и полноты, чем только keywords или только semantics.</p></div></div>
            </div>
        """,
    },
    {
        "title": "Reranking",
        "eyebrow": "Ranking Layer",
        "html": """
            <div class="grid two">
              <div class="card"><h3>Какие сигналы учитываются</h3><ul><li>lexical relevance;</li><li>semantic similarity;</li><li>topical relevance;</li><li>штрафы за boilerplate и технический шум;</li><li>anchor-term logic для узких запросов.</li></ul></div>
              <div class="card accent"><h3>Model-based reranker</h3><div class="chips"><span>BAAI/bge-reranker-v2-m3</span></div><p>Модель получает пару «запрос + chunk» и оценивает, насколько этот фрагмент действительно отвечает на запрос.</p></div>
            </div>
            <blockquote>Retrieval находит кандидатов, а reranker убирает шум, точнее сортирует выдачу и повышает качество контекста для LLM.</blockquote>
        """,
    },
    {
        "title": "Генерация аналитической сводки",
        "eyebrow": "Generation",
        "html": """
            <div class="grid two">
              <div class="card"><h3>Как строится ответ</h3><p>Когда лучшие chunks найдены и отранжированы, система передаёт их в языковую модель для генерации сводки.</p><div class="chips"><span>qwen2.5</span><span>3b / 7b</span><span>Ollama</span></div></div>
              <div class="card"><h3>Роль LLM</h3><ul><li>retrieval отвечает за поиск;</li><li>reranker отвечает за порядок и точность;</li><li>LLM отвечает за формирование понятного ответа.</li></ul></div>
            </div>
            <div class="card wide"><h3>Почему выбран Qwen</h3><ul><li>хорошая совместимость с локальным запуском через Ollama;</li><li>приемлемое качество для summarization и structured extraction;</li><li>удобный баланс между качеством и вычислительной стоимостью.</li></ul></div>
        """,
    },
    {
        "title": "Topic Discovery",
        "eyebrow": "Corpus Analytics",
        "html": """
            <div class="grid two">
              <div class="card"><h3>Методы</h3><div class="chips"><span>BERTopic</span><span>HDBSCAN</span><span>UMAP</span><span>c-TF-IDF</span></div><ul><li>группировка статей по близости;</li><li>понижение размерности признаков;</li><li>выделение плотных кластеров;</li><li>извлечение ключевых слов темы.</li></ul></div>
              <div class="card"><h3>Зачем это нужно</h3><ul><li>разбить корпус на более однородные смысловые области;</li><li>подготовить почву для narrative analysis;</li><li>ограничить смешивание слишком разных сюжетов.</li></ul></div>
            </div>
        """,
    },
    {
        "title": "Narrative Extraction",
        "eyebrow": "Narrative Layer",
        "html": """
            <div class="grid two">
              <div class="card accent"><h3>Какая модель используется</h3><div class="chips"><span>qwen2.5</span></div><p>Внутри корпуса или тематических массивов система пытается извлечь narrative frames на уровне статей.</p></div>
              <div class="card"><h3>Что включает narrative frame</h3><ul><li>основное утверждение;</li><li>ключевых акторов;</li><li>причину;</li><li>механизм;</li><li>последствия;</li><li>ожидание будущего;</li><li>тональность и импликации.</li></ul></div>
            </div>
            <blockquote>Это переход от уровня «о чём статья» к уровню «какую интерпретационную историю она строит».</blockquote>
        """,
    },
    {
        "title": "Narrative Embeddings и Clustering",
        "eyebrow": "Narrative Structure",
        "html": """
            <div class="grid two">
              <div class="card"><h3>Почему narrative embeddings отдельные</h3><ul><li>близость статей не равна близости нарративов;</li><li>разные события могут поддерживать один и тот же narrative;</li><li>одна тема может включать несколько конкурирующих narratives.</li></ul></div>
              <div class="card accent"><h3>Кластеризация</h3><div class="chips"><span>HDBSCAN</span></div><ul><li>работает с плотностной кластеризацией;</li><li>не заставляет каждый объект входить в кластер;</li><li>оставляет шум как noise.</li></ul></div>
            </div>
        """,
    },
    {
        "title": "Narrative Labeling и динамика",
        "eyebrow": "Dynamics",
        "html": """
            <div class="grid two">
              <div class="card"><h3>После кластеризации</h3><p>Система строит labels для narrative clusters. Для этого снова используется qwen2.5.</p><ul><li>название нарратива;</li><li>краткое описание;</li><li>canonical claim;</li><li>ключевые акторы;</li><li>причинная цепочка;</li><li>доминирующая тональность;</li><li>возможный контр-нарратив.</li></ul></div>
              <div class="card"><h3>Динамика</h3><ul><li>частота;</li><li>доля в потоке;</li><li>рост и спад;</li><li>устойчивость;</li><li>распределение по источникам;</li><li>всплески во времени.</li></ul><p class="footnote">Narratives становятся полноценными наблюдаемыми объектами анализа.</p></div>
            </div>
        """,
    },
    {
        "title": "Связь компонентов между собой",
        "eyebrow": "System Flow",
        "html": """
            <div class="steps vertical">
              <div class="step"><span>1</span><div><p>Новости загружаются и нормализуются.</p></div></div>
              <div class="step"><span>2</span><div><p>Сохраняются в SQLite.</p></div></div>
              <div class="step"><span>3</span><div><p>Статьи режутся на чанки, для чанков строятся embeddings.</p></div></div>
              <div class="step"><span>4</span><div><p>Retrieval ищет кандидатов, reranker улучшает ранжирование, LLM формирует итоговую сводку.</p></div></div>
              <div class="step"><span>5</span><div><p>Topic discovery группирует корпус, LLM извлекает narrative frames, narratives эмбеддятся и кластеризуются.</p></div></div>
              <div class="step"><span>6</span><div><p>Narratives получают labels и временную динамику, результаты показываются в интерфейсе.</p></div></div>
            </div>
            <blockquote>Каждый следующий слой не заменяет предыдущий, а строится поверх него: без корпуса нет retrieval, без retrieval нет качественного RAG, без topic/narrative-слоя нет глубокой аналитики.</blockquote>
        """,
    },
    {
        "title": "Почему выбран именно такой стек",
        "eyebrow": "Tech Decisions",
        "html": """
            <div class="grid three">
              <div class="card"><h3>SQLite</h3><p>Быстрый старт, локальная разработка, низкая инфраструктурная стоимость, достаточность для текущего этапа.</p></div>
              <div class="card"><h3>Chunk-based architecture</h3><p>Выше точность retrieval и лучше контроль над контекстом.</p></div>
              <div class="card"><h3>Embeddings</h3><p>Нужны для semantic search и работы с перефразировками.</p></div>
              <div class="card"><h3>Hybrid retrieval</h3><p>Сочетает точность keywords и гибкость semantics.</p></div>
              <div class="card"><h3>Reranker</h3><p>Улучшает порядок кандидатов и качество итогового контекста.</p></div>
              <div class="card"><h3>Qwen</h3><p>Подходит для локального LLM-контура и работает как summarizer, extractor и labeler.</p></div>
              <div class="card"><h3>BERTopic + HDBSCAN + UMAP</h3><p>Даёт понятный и модульный pipeline для topic discovery.</p></div>
              <div class="card"><h3>Narrative-level embeddings</h3><p>Позволяют работать с интерпретациями, а не только со статьями.</p></div>
              <div class="card"><h3>Ollama</h3><p>Упрощает локальный запуск моделей и снижает зависимость от внешних облачных API.</p></div>
            </div>
        """,
    },
    {
        "title": "Итог",
        "eyebrow": "Conclusion",
        "html": """
            <p class="lead">News Intelligence — это локальная AI-платформа для анализа новостного потока, которая объединяет сбор и хранение новостей, смысловой поиск, retrieval-augmented generation, topic analysis и narrative intelligence.</p>
            <div class="grid two">
              <div class="card accent"><h3>Главная ценность</h3><p>Мы сокращаем путь от большого и шумного новостного массива к понятной аналитической картине.</p></div>
              <div class="card"><h3>Что получает пользователь</h3><ul><li>понимание, какие темы обсуждаются;</li><li>понимание, какие интерпретации повторяются;</li><li>видимость того, какие нарративы усиливаются;</li><li>представление о том, как меняется информационное поле во времени.</li></ul></div>
            </div>
            <blockquote>News Intelligence помогает не просто находить новости, а понимать структуру информационной среды.</blockquote>
        """,
    },
]


css = """
:root {
  --bg: #f6f7fb;
  --surface: rgba(255,255,255,0.76);
  --surface-strong: rgba(255,255,255,0.92);
  --text: #111318;
  --muted: #697386;
  --line: rgba(17, 19, 24, 0.08);
  --accent: #0071e3;
  --accent-soft: rgba(0,113,227,0.08);
  --shadow: 0 24px 80px rgba(17, 19, 24, 0.08);
  --radius: 34px;
  --font-display: "SF Pro Display", "Segoe UI", Inter, system-ui, sans-serif;
  --font-body: "SF Pro Text", "Segoe UI", Inter, system-ui, sans-serif;
}
* { box-sizing: border-box; }
html, body {
  margin: 0;
  min-height: 100%;
  background:
    radial-gradient(circle at top center, rgba(0,113,227,0.10), transparent 28%),
    radial-gradient(circle at bottom left, rgba(52,199,89,0.08), transparent 22%),
    linear-gradient(180deg, #fbfbfd 0%, #f2f4f8 100%);
  color: var(--text);
  font-family: var(--font-body);
  overflow: hidden;
}
body::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background:
    linear-gradient(90deg, rgba(17,19,24,0.018) 1px, transparent 1px),
    linear-gradient(180deg, rgba(17,19,24,0.018) 1px, transparent 1px);
  background-size: 56px 56px;
  mask-image: radial-gradient(circle at center, black 30%, transparent 78%);
}
.deck { min-height: 100vh; padding: 18px 20px 84px; position: relative; }
.topbar {
  max-width: 1320px; margin: 0 auto 18px; padding: 10px 14px 10px 18px;
  display: flex; justify-content: space-between; align-items: center; gap: 16px;
  border: 1px solid var(--line); border-radius: 999px;
  background: rgba(255,255,255,0.72); backdrop-filter: blur(28px);
  box-shadow: 0 10px 30px rgba(17,19,24,0.04);
}
.brand { display:flex; align-items:center; gap:12px; min-width:0; }
.brand-mark {
  width: 28px; height: 28px; border-radius: 50%;
  background: linear-gradient(135deg, #0f172a, #0071e3);
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.2);
  flex: 0 0 auto;
}
.brand strong { display:block; font-size:0.96rem; }
.brand span { display:block; color:var(--muted); font-size:0.82rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.meta { display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }
.meta span, .hero-chips span, .chips span {
  padding: 9px 13px; border-radius: 999px; border: 1px solid var(--line);
  background: rgba(255,255,255,0.7); color: var(--muted); font-size: 0.84rem;
}
.viewport { max-width: 1320px; margin: 0 auto; position: relative; }
.slides { position: relative; height: calc(100vh - 162px); }
.slide {
  position:absolute; inset:0; opacity:0; pointer-events:none;
  transform: translateY(18px) scale(0.992);
  transition: opacity 240ms ease, transform 240ms ease;
  display:grid; grid-template-rows:auto 1fr; gap:18px;
}
.slide.active { opacity:1; pointer-events:auto; transform:translateY(0) scale(1); }
.slide-header { display:flex; justify-content:space-between; align-items:flex-end; gap:24px; padding:0 6px; }
.eyebrow {
  display:inline-flex; align-items:center; gap:8px; margin-bottom:14px;
  padding:7px 12px; border-radius:999px; background: var(--accent-soft);
  border:1px solid rgba(0,113,227,0.10); color:var(--accent);
  font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.14em;
}
.slide-header h2 {
  margin:0; max-width:940px; font-family:var(--font-display);
  font-size: clamp(2.2rem, 3.2vw, 4rem); line-height:0.94; letter-spacing:-0.05em;
}
.num { min-width:72px; text-align:right; color:rgba(17,19,24,0.14); font-size:3.5rem; font-weight:800; letter-spacing:-0.06em; }
.panel {
  min-height:0; overflow:auto; padding:34px;
  border:1px solid var(--line); border-radius:var(--radius);
  background: linear-gradient(180deg, var(--surface), var(--surface-strong));
  backdrop-filter: blur(30px); box-shadow: var(--shadow);
}
.hero .slide-header { display:none; }
.hero .panel {
  display:flex; flex-direction:column; justify-content:center; padding:56px;
  background:
    radial-gradient(circle at top right, rgba(0,113,227,0.12), transparent 24%),
    linear-gradient(180deg, rgba(255,255,255,0.84), rgba(255,255,255,0.96));
}
.hero-mark {
  display:inline-flex; align-items:center; gap:10px; color:var(--muted);
  font-size:0.9rem; text-transform:uppercase; letter-spacing:0.08em;
}
.hero-mark::before { content:""; width:46px; height:1px; background:rgba(17,19,24,0.18); }
.hero-title {
  margin:20px 0 0; font-family:var(--font-display);
  font-size: clamp(3.7rem, 8vw, 7.2rem); line-height:0.9; letter-spacing:-0.08em;
}
.hero-subtitle {
  margin:22px 0 0; max-width:860px; color:#40495a;
  font-size: clamp(1.12rem, 2vw, 1.42rem); line-height:1.66;
}
.hero-chips { display:flex; flex-wrap:wrap; gap:10px; margin-top:30px; }
.team-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; margin-top:30px; }
.person {
  padding:18px; border-radius:24px; border:1px solid var(--line);
  background: rgba(255,255,255,0.58);
}
.person strong { display:block; margin-bottom:6px; }
.person span { color:var(--muted); line-height:1.45; }
.lead { max-width:980px; color:#2d3440; font-size:1.18rem; line-height:1.72; margin:0 0 10px; }
.grid { display:grid; gap:18px; margin-top:24px; }
.grid.two { grid-template-columns:repeat(2,minmax(0,1fr)); }
.grid.three { grid-template-columns:repeat(3,minmax(0,1fr)); }
.card {
  padding:22px; border-radius:26px; border:1px solid rgba(17,19,24,0.06);
  background: rgba(255,255,255,0.62); box-shadow: 0 10px 30px rgba(17,19,24,0.04);
}
.card.accent { background: linear-gradient(180deg, rgba(0,113,227,0.07), rgba(255,255,255,0.74)); }
.card.wide { margin-top:20px; }
.card h3 { margin:0 0 12px; font-size:1.04rem; }
.card p, .card li { margin:0; color:var(--muted); line-height:1.62; }
ul { margin:10px 0 0; padding-left:20px; }
li { margin:7px 0; }
.cols { columns: 2; column-gap: 28px; }
.metrics { display:grid; gap:16px; margin-top:24px; }
.metrics.four { grid-template-columns:repeat(4,minmax(0,1fr)); }
.metrics > div {
  padding:18px; border-radius:24px; border:1px solid var(--line);
  background: rgba(255,255,255,0.62);
}
.metrics strong { display:block; font-size:2rem; margin-bottom:8px; }
.metrics span { color:var(--muted); line-height:1.45; }
.note-band { display:flex; flex-wrap:wrap; gap:10px; margin-top:22px; }
.note-band span, .footnote {
  padding:10px 14px; border-radius:999px; background:rgba(17,19,24,0.04); border:1px solid var(--line); color:var(--muted);
}
blockquote {
  margin:24px 0 0; padding:24px 26px; border-radius:28px;
  background: linear-gradient(180deg, rgba(0,113,227,0.06), rgba(255,255,255,0.65));
  border:1px solid rgba(0,113,227,0.10); color:#253041; line-height:1.7;
}
.steps { display:grid; gap:14px; margin-top:24px; }
.step { display:grid; grid-template-columns:68px 1fr; gap:16px; align-items:start; }
.step span {
  width:54px; height:54px; border-radius:18px; display:grid; place-items:center; font-weight:800; color:var(--accent);
  background: rgba(0,113,227,0.08); border:1px solid rgba(0,113,227,0.12);
}
.step h3 { margin:0 0 8px; }
.step p { margin:0; color:var(--muted); line-height:1.58; }
.footer {
  position:absolute; left:0; right:0; bottom:18px; max-width:1320px; margin:0 auto;
  display:flex; justify-content:space-between; align-items:center; gap:18px; padding:0 6px;
}
.nav { display:flex; gap:12px; }
button.nav-btn {
  border:1px solid var(--line); background: rgba(255,255,255,0.72); color:var(--text);
  padding:12px 16px; border-radius:14px; cursor:pointer; font:inherit; backdrop-filter: blur(24px);
}
button.nav-btn:hover { background: rgba(255,255,255,0.96); }
.progress { flex:1; display:flex; justify-content:center; gap:6px; }
.dot { width:10px; height:10px; border-radius:999px; background: rgba(17,19,24,0.12); transition:180ms ease; }
.dot.active { width:28px; background: linear-gradient(90deg, var(--accent), #6ea8ff); }
@media (max-width: 1180px) {
  .deck { padding:14px 14px 82px; }
  .slides { height: calc(100vh - 156px); }
  .grid.two, .grid.three, .team-grid, .metrics.four { grid-template-columns:1fr; }
  .cols { columns: 1; }
  .meta { display:none; }
  .hero .panel { padding:36px 28px; }
  .hero-title { font-size: clamp(3rem, 14vw, 5rem); }
}
@media print {
  body { overflow: visible; background:#fff; }
  .topbar, .footer { display:none; }
  .viewport, .slides, .slide, .slide.active { position:static; height:auto; opacity:1; transform:none; overflow:visible; page-break-after:always; }
  .panel, .card, .person, .metrics > div { box-shadow:none; background:#fff; }
}
"""


def render_slide(index: int, slide: dict[str, str]) -> str:
    active = " active" if index == 1 else ""
    hero = " hero" if slide.get("kind") == "hero" else ""
    return f"""
        <section class="slide{hero}{active}">
          <div class="slide-header">
            <div>
              <div class="eyebrow">{slide["eyebrow"]}</div>
              <h2>{slide["title"]}</h2>
            </div>
            <div class="num">{index:02d}</div>
          </div>
          <div class="panel">
            {slide["html"]}
          </div>
        </section>
    """


slides_html = "\n".join(render_slide(i, slide) for i, slide in enumerate(slides, start=1))

html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>News Intelligence — Clean Deck</title>
  <style>{css}</style>
</head>
<body>
  <div class="deck">
    <div class="topbar">
      <div class="brand">
        <div class="brand-mark"></div>
        <div>
          <strong>News Intelligence</strong>
          <span>AI-платформа для смыслового поиска, аналитики и выявления нарративов</span>
        </div>
      </div>
      <div class="meta">
        <span>24 слайда</span>
        <span>Product + Architecture</span>
        <span>HTML Deck</span>
      </div>
    </div>

    <main class="viewport">
      <div class="slides">
        {slides_html}
      </div>
      <div class="footer">
        <div class="nav">
          <button class="nav-btn" id="prevBtn" type="button">← Назад</button>
          <button class="nav-btn" id="nextBtn" type="button">Вперёд →</button>
        </div>
        <div class="progress" id="progress"></div>
      </div>
    </main>
  </div>

  <script>
    const slides = Array.from(document.querySelectorAll('.slide'));
    const progress = document.getElementById('progress');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    let currentIndex = 0;

    function buildNavigation() {{
      slides.forEach((slide, index) => {{
        const dot = document.createElement('div');
        dot.className = 'dot';
        dot.addEventListener('click', () => setSlide(index));
        progress.appendChild(dot);
      }});
    }}

    function setSlide(index) {{
      currentIndex = Math.max(0, Math.min(index, slides.length - 1));
      slides.forEach((slide, idx) => slide.classList.toggle('active', idx === currentIndex));
      Array.from(progress.children).forEach((dot, idx) => dot.classList.toggle('active', idx === currentIndex));
      prevBtn.disabled = currentIndex === 0;
      nextBtn.disabled = currentIndex === slides.length - 1;
    }}

    prevBtn.addEventListener('click', () => setSlide(currentIndex - 1));
    nextBtn.addEventListener('click', () => setSlide(currentIndex + 1));
    window.addEventListener('keydown', (event) => {{
      if (event.key === 'ArrowRight' || event.key === 'PageDown') setSlide(currentIndex + 1);
      if (event.key === 'ArrowLeft' || event.key === 'PageUp') setSlide(currentIndex - 1);
      if (event.key === 'Home') setSlide(0);
      if (event.key === 'End') setSlide(slides.length - 1);
    }});

    buildNavigation();
    setSlide(0);
  </script>
</body>
</html>
"""


out = Path(__file__).with_name("news_intelligence_presentation_apple.html")
out.write_text(html, encoding="utf-8")
print(out)
