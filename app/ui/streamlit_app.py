from __future__ import annotations

from datetime import date

import streamlit as st

from app.bootstrap import build_app_services
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


def render_rag_tab(services) -> None:
    st.subheader("Поиск по новостям (RAG)")
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
            st.error(f"Ошибка RAG-запроса: {exc}")
            return

        if not result.source_articles and not result.top_chunks:
            st.info("За выбранный период релевантных данных не найдено.")
            return

        st.markdown("### Сводка")
        st.write(result.summary_text or "Сводка недоступна.")

        st.markdown("### Исходные статьи")
        if result.source_articles:
            for article in result.source_articles:
                st.markdown(f"- [{article.title}]({article.url})")
        else:
            st.info("Поддерживающие статьи не найдены.")

        with st.expander("Релевантные фрагменты"):
            if result.top_chunks:
                for chunk in result.top_chunks:
                    st.markdown(f"**{chunk.article_title}**")
                    st.write(chunk.chunk_text)
            else:
                st.info("Фрагменты не найдены.")


def render_narratives_tab(services) -> None:
    st.subheader("Анализ нарративов")
    run_mode = st.radio(
        "Режим",
        options=("По теме", "По всему корпусу"),
        horizontal=True,
        key="narrative_mode",
    )
    topic_text = ""
    if run_mode == "По теме":
        topic_text = st.text_input("Тема", key="narrative_topic")
    source_options = get_source_options(services)
    selected_source_label = st.selectbox("Источник", list(source_options.keys()), key="narrative_source")
    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("Дата с", value=date(2026, 4, 1), key="narrative_date_from")
    with col2:
        date_to = st.date_input("Дата по", value=date.today(), key="narrative_date_to")

    if st.button("Запустить нарративы", key="narrative_run"):
        if run_mode == "По теме" and not topic_text.strip():
            st.warning("Введите тему.")
            return
        try:
            result = services.narrative_run_service.run(
                topic_text=topic_text.strip() if run_mode == "По теме" else None,
                date_from=f"{date_from.isoformat()}T00:00:00",
                date_to=f"{date_to.isoformat()}T23:59:59",
                source_domains=source_options[selected_source_label],
            )
        except Exception as exc:
            st.error(f"Ошибка запуска narratives: {exc}")
            return

        narrative_results = result["results"]
        if not narrative_results:
            if run_mode == "По теме":
                st.info("За выбранную тему и период нарративы не найдены.")
            else:
                st.info("За выбранный период нарративы не найдены.")
            return

        st.markdown("### Результаты")
        by_type = {item.narrative_type: item for item in narrative_results}
        for narrative_type in ("predictive", "causal", "meta"):
            item = by_type.get(narrative_type)
            type_titles = {
                "predictive": "Прогнозный",
                "causal": "Причинно-следственный",
                "meta": "Мета-нарратив",
            }
            st.markdown(f"## {type_titles.get(narrative_type, narrative_type)}")
            if item is None:
                st.info("Нарратив этого типа не найден.")
                continue

            st.markdown(f"**{item.title}**")
            st.write(item.formulation)
            st.caption(item.explanation or "")
            st.caption(f"strength_score={item.strength_score}")

            articles = services.narrative_result_repository.list_articles_for_result(item.id)
            if articles:
                st.markdown("Поддерживающие статьи")
                for article in articles[:5]:
                    st.markdown(f"- [{article.title}]({article.url})")
            else:
                st.info("Поддерживающие статьи не сохранены.")


def main() -> None:
    st.set_page_config(page_title="NarrativeNews", layout="wide")
    st.title("NarrativeNews")
    st.caption("Локальный интерфейс для RAG-поиска и анализа нарративов.")

    try:
        _connection, services, settings = get_connection_and_services()
    except Exception as exc:
        st.error(f"Не удалось инициализировать приложение: {exc}")
        return

    st.sidebar.markdown("### База данных")
    st.sidebar.code(str(settings.database_path))
    st.sidebar.markdown("### LLM")
    st.sidebar.code(f"{settings.llm_provider} | {settings.llm_model_name}")

    rag_tab, narratives_tab = st.tabs(["RAG", "Нарративы"])
    with rag_tab:
        render_rag_tab(services)
    with narratives_tab:
        render_narratives_tab(services)


if __name__ == "__main__":
    main()
