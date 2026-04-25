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


def main() -> None:
    st.set_page_config(page_title="News RAG", layout="wide")
    st.title("News RAG")
    st.caption("Локальный интерфейс для поиска по новостному корпусу.")

    try:
        _connection, services, settings = get_connection_and_services()
    except Exception as exc:
        st.error(f"Не удалось инициализировать приложение: {exc}")
        return

    st.sidebar.markdown("### База данных")
    st.sidebar.code(str(settings.database_path))

    render_rag(services)


if __name__ == "__main__":
    main()
