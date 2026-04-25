from __future__ import annotations

from datetime import date

import streamlit as st

from app.bootstrap import build_app_services, build_narrative_intelligence_services
from app.config.settings import get_settings
from app.db.connection import create_connection
from app.db.init_db import initialize_database


@st.cache_resource
def get_connection_and_services():
    settings = get_settings()
    initialize_database(settings.database_path)
    connection = create_connection(settings.database_path)
    services = build_app_services(connection, settings)
    return connection, services, settings


def get_source_options(services) -> dict[str, list[str] | None]:
    options: dict[str, list[str] | None] = {"Все источники": None}
    for source in services.source_repository.list():
        label = source.name
        if source.domain == "ria.ru":
            label = "РИА Новости"
        elif source.domain == "lenta.ru":
            label = "Лента.ру"
        options[label] = [source.domain]
    return options


def render_rag(services) -> None:
    st.subheader("Поиск по новостям")
    query = st.text_input("Запрос", key="rag_query")
    source_options = get_source_options(services)
    selected_source_label = st.selectbox("Источник", list(source_options.keys()), key="rag_source")
    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("Дата с", value=date(2026, 4, 1), key="rag_date_from")
    with col2:
        date_to = st.date_input("Дата по", value=date.today(), key="rag_date_to")

    if st.button("Запустить поиск", key="rag_run"):
        if not query.strip():
            st.warning("Введите запрос.")
            return
        try:
            result = services.rag_service.answer(
                query=query.strip(),
                date_from=f"{date_from.isoformat()}T00:00:00",
                date_to=f"{date_to.isoformat()}T23:59:59",
                limit=8,
                include_debug_chunks=True,
                source_domains=source_options[selected_source_label],
            )
        except Exception as exc:
            st.error(f"Ошибка RAG-поиска: {exc}")
            return

        if not result.source_articles and not result.top_chunks:
            st.info("За выбранный период релевантные данные не найдены.")
            return

        st.markdown("### Сводка")
        st.write(result.summary_text or "Сводка не сформирована.")

        st.markdown("### Исходные статьи")
        if result.source_articles:
            for article in result.source_articles:
                st.markdown(f"- [{article.title}]({article.url})")
        else:
            st.info("Подходящие статьи не найдены.")

        with st.expander("Релевантные фрагменты"):
            if result.top_chunks:
                for chunk in result.top_chunks:
                    st.markdown(f"**{chunk.article_title}**")
                    st.write(chunk.chunk_text)
            else:
                st.info("Фрагменты не найдены.")


def build_narrative_pipeline(connection, settings):
    return build_narrative_intelligence_services(connection, settings)


def _format_keywords(keywords: tuple[str, ...]) -> str:
    return ", ".join(keyword for keyword in keywords if keyword) or "—"


def _render_topics(result) -> None:
    st.markdown("### Темы")
    if not result.topics:
        st.info("Темы не выделены.")
        return
    for topic in result.topics[:12]:
        with st.expander(f"{topic.label} ({len(topic.article_ids)} статей)"):
            st.write(f"**ID:** {topic.topic_id}")
            st.write(f"**Ключевые слова:** {_format_keywords(topic.keywords)}")


def _render_clusters(result) -> None:
    st.markdown("### Нарративные кластеры")
    if not result.clusters:
        st.info("Устойчивые нарративы не выделены.")
        return

    labels_by_cluster_id = {label.cluster_id: label for label in result.labels}
    assignments_by_cluster_id: dict[str, int] = {}
    for assignment in result.assignments:
        if assignment.assigned and assignment.cluster_id:
            assignments_by_cluster_id[assignment.cluster_id] = assignments_by_cluster_id.get(assignment.cluster_id, 0) + 1

    ordered_clusters = sorted(
        result.clusters,
        key=lambda cluster: assignments_by_cluster_id.get(cluster.cluster_id, len(cluster.frame_ids)),
        reverse=True,
    )

    for cluster in ordered_clusters[:20]:
        label = labels_by_cluster_id.get(cluster.cluster_id)
        title = label.title if label and label.title else cluster.cluster_id
        suffix = " [noise]" if cluster.noise else ""
        with st.expander(f"{title}{suffix}"):
            st.write(f"**Cluster ID:** {cluster.cluster_id}")
            st.write(f"**Тема:** {cluster.topic_id or '—'}")
            st.write(f"**Фреймов в кластере:** {len(cluster.frame_ids)}")
            st.write(f"**Назначений:** {assignments_by_cluster_id.get(cluster.cluster_id, 0)}")
            if label is None:
                st.info("LLM-лейбл для кластера не сформирован.")
                continue
            st.write(f"**Описание:** {label.summary or '—'}")
            st.write(f"**Canonical claim:** {label.canonical_claim or '—'}")
            st.write(f"**Ключевые акторы:** {', '.join(label.key_actors) if label.key_actors else '—'}")
            st.write(f"**Причинная цепочка:** {' → '.join(label.causal_chain) if label.causal_chain else '—'}")
            st.write(f"**Доминирующая тональность:** {label.dominant_tone or '—'}")
            st.write(f"**Контр-нарратив:** {label.counter_narrative or '—'}")
            if label.typical_formulations:
                st.write("**Типичные формулировки:**")
                for item in label.typical_formulations[:5]:
                    st.markdown(f"- {item}")
            if label.representative_examples:
                st.write("**Representative examples:**")
                for item in label.representative_examples[:5]:
                    st.markdown(f"- {item}")


def _render_dynamics(result) -> None:
    st.markdown("### Динамика")
    if not result.dynamics:
        st.info("Данные по динамике пока не выделены.")
        return

    labels_by_cluster_id = {label.cluster_id: label for label in result.labels}
    for series in result.dynamics[:12]:
        label = labels_by_cluster_id.get(series.cluster_id)
        title = label.title if label and label.title else series.cluster_id
        with st.expander(title):
            st.write(f"**Всего статей:** {series.total_articles}")
            st.write(
                f"**Рост:** {round(series.growth_rate, 3) if series.growth_rate is not None else '—'}"
            )
            st.write(
                f"**Устойчивость:** {round(series.stability_score, 3) if series.stability_score is not None else '—'}"
            )
            for point in series.points:
                st.markdown(
                    f"- `{point.period_start}`: статей={point.article_count}, "
                    f"доля={round(point.share_of_corpus, 3)}, "
                    f"diversity={round(point.source_diversity, 3)}, "
                    f"burst={round(point.burst_score, 3) if point.burst_score is not None else '—'}"
                )


def render_narratives(connection, services, settings) -> None:
    st.subheader("Narrative Intelligence")
    st.caption(
        "Этот режим ищет не просто темы, а повторяющиеся интерпретационные истории: "
        "кто действует, почему это происходит, к чему ведёт и какое ожидание формируется."
    )

    source_options = get_source_options(services)
    selected_source_label = st.selectbox(
        "Источник",
        list(source_options.keys()),
        key="narrative_source",
    )
    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("Дата с", value=date(2026, 1, 1), key="narrative_date_from")
    with col2:
        date_to = st.date_input("Дата по", value=date.today(), key="narrative_date_to")

    if st.button("Запустить анализ нарративов", key="narrative_run"):
        try:
            pipeline = build_narrative_pipeline(connection, settings)
        except Exception as exc:
            st.error(f"Не удалось подготовить narrative pipeline: {exc}")
            return

        try:
            with st.spinner("Анализируем корпус, выделяем темы, извлекаем фреймы и собираем нарративы..."):
                result = pipeline.run(
                    date_from=f"{date_from.isoformat()}T00:00:00",
                    date_to=f"{date_to.isoformat()}T23:59:59",
                    source_domains=source_options[selected_source_label],
                )
        except Exception as exc:
            st.error(f"Ошибка при запуске narrative intelligence: {exc}")
            return

        st.markdown("### Сводка запуска")
        col1, col2, col3 = st.columns(3)
        col1.metric("Документы", len(result.documents))
        col2.metric("Темы", len(result.topics))
        col3.metric("Фреймы", len(result.frames))
        col4, col5, col6 = st.columns(3)
        col4.metric("Кластеры", len(result.clusters))
        col5.metric("Лейблы", len(result.labels))
        assigned_count = sum(1 for assignment in result.assignments if assignment.assigned)
        col6.metric("Назначения", assigned_count)

        no_clear_count = sum(1 for frame in result.frames if frame.status == "no_clear_narrative")
        if no_clear_count:
            st.info(f"Статей без явного нарратива: {no_clear_count}")

        _render_topics(result)
        _render_clusters(result)
        _render_dynamics(result)

        if result.evaluation is not None and result.evaluation.notes:
            with st.expander("Оценка качества"):
                for note in result.evaluation.notes:
                    st.markdown(f"- {note}")


def main() -> None:
    st.set_page_config(page_title="News Intelligence", layout="wide")
    st.title("News Intelligence")
    st.caption("Локальный интерфейс для анализа новостного корпуса: RAG и narrative intelligence.")

    try:
        connection, services, settings = get_connection_and_services()
    except Exception as exc:
        st.error(f"Не удалось инициализировать приложение: {exc}")
        return

    st.sidebar.markdown("### База данных")
    st.sidebar.code(str(settings.database_path))

    rag_tab, narrative_tab = st.tabs(["RAG", "Нарративы"])
    with rag_tab:
        render_rag(services)
    with narrative_tab:
        render_narratives(connection, services, settings)


if __name__ == "__main__":
    main()
