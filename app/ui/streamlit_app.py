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


def render_rag_tab(services) -> None:
    st.subheader("RAG Search")
    query = st.text_input("Query", key="rag_query")
    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("Date From", value=date(2026, 4, 1), key="rag_date_from")
    with col2:
        date_to = st.date_input("Date To", value=date.today(), key="rag_date_to")

    if st.button("Run RAG", key="rag_run"):
        if not query.strip():
            st.warning("Enter a query first.")
            return
        try:
            result = services.rag_service.answer(
                query=query.strip(),
                date_from=f"{date_from.isoformat()}T00:00:00",
                date_to=f"{date_to.isoformat()}T23:59:59",
                limit=8,
                include_debug_chunks=True,
            )
        except Exception as exc:
            st.error(f"RAG request failed: {exc}")
            return

        if not result.source_articles and not result.top_chunks:
            st.info("No relevant data found for the selected period.")
            return

        st.markdown("### Summary")
        st.write(result.summary_text or "No summary available.")

        st.markdown("### Source Articles")
        if result.source_articles:
            for article in result.source_articles:
                st.markdown(f"- [{article.title}]({article.url})")
        else:
            st.info("No source articles found.")

        with st.expander("Top Chunks"):
            if result.top_chunks:
                for chunk in result.top_chunks:
                    st.markdown(f"**{chunk.article_title}**")
                    st.write(chunk.chunk_text)
            else:
                st.info("No chunks found.")


def render_narratives_tab(services) -> None:
    st.subheader("Narrative Run")
    topic_text = st.text_input("Topic", key="narrative_topic")
    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("Date From", value=date(2026, 4, 1), key="narrative_date_from")
    with col2:
        date_to = st.date_input("Date To", value=date.today(), key="narrative_date_to")

    if st.button("Run Narratives", key="narrative_run"):
        if not topic_text.strip():
            st.warning("Enter a topic first.")
            return
        try:
            result = services.narrative_run_service.run(
                topic_text=topic_text.strip(),
                date_from=f"{date_from.isoformat()}T00:00:00",
                date_to=f"{date_to.isoformat()}T23:59:59",
            )
        except Exception as exc:
            st.error(f"Narrative run failed: {exc}")
            return

        narrative_results = result["results"]
        if not narrative_results:
            st.info("No narrative results found for the selected topic and period.")
            return

        st.markdown("### Narrative Results")
        by_type = {item.narrative_type: item for item in narrative_results}
        for narrative_type in ("predictive", "causal", "meta"):
            item = by_type.get(narrative_type)
            st.markdown(f"## {narrative_type.title()}")
            if item is None:
                st.info(f"No {narrative_type} narrative found.")
                continue

            st.markdown(f"**{item.title}**")
            st.write(item.formulation)
            st.caption(item.explanation or "")
            st.caption(f"strength_score={item.strength_score}")

            articles = services.narrative_result_repository.list_articles_for_result(item.id)
            if articles:
                st.markdown("Supporting Articles")
                for article in articles[:5]:
                    st.markdown(f"- [{article.title}]({article.url})")
            else:
                st.info("No supporting articles saved.")


def main() -> None:
    st.set_page_config(page_title="NarrativeNews MVP", layout="wide")
    st.title("NarrativeNews MVP")
    st.caption("Minimal Streamlit UI for RAG and narrative exploration.")

    try:
        _connection, services, settings = get_connection_and_services()
    except Exception as exc:
        st.error(f"Failed to initialize app: {exc}")
        return

    st.sidebar.markdown("### Database")
    st.sidebar.code(str(settings.database_path))
    st.sidebar.markdown("### LLM")
    st.sidebar.code(f"{settings.llm_provider} | {settings.llm_model_name}")

    rag_tab, narratives_tab = st.tabs(["RAG", "Narratives"])
    with rag_tab:
        render_rag_tab(services)
    with narratives_tab:
        render_narratives_tab(services)


if __name__ == "__main__":
    main()
