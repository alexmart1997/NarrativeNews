from __future__ import annotations

from datetime import date

import streamlit as st

from app.bootstrap import build_app_services
from app.config.settings import get_settings
from app.db.connection import create_connection
from app.db.init_db import initialize_database
from app.services import build_source_domains_key


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


def _format_keywords(keywords: list[str] | tuple[str, ...]) -> str:
    return ", ".join(keyword for keyword in keywords if keyword) or "—"


def _render_topics(topics: list[dict[str, object]]) -> None:
    st.markdown("### Темы")
    if not topics:
        st.info("Темы не выделены.")
        return
    for topic in topics[:12]:
        article_ids = topic.get("article_ids") or []
        keywords = topic.get("keywords") or []
        with st.expander(f"{topic.get('label', 'topic')} ({len(article_ids)} статей)"):
            st.write(f"**ID:** {topic.get('topic_id', '—')}")
            st.write(f"**Ключевые слова:** {_format_keywords(keywords)}")


def _render_clusters(snapshot: dict[str, object]) -> None:
    st.markdown("### Нарративные кластеры")
    clusters = list(snapshot.get("clusters") or [])
    labels = list(snapshot.get("labels") or [])
    assignments = list(snapshot.get("assignments") or [])
    if not clusters:
        st.info("Устойчивые нарративы не выделены.")
        return

    labels_by_cluster_id = {
        str(label.get("cluster_id")): label
        for label in labels
        if isinstance(label, dict) and label.get("cluster_id") is not None
    }
    assignments_by_cluster_id: dict[str, int] = {}
    for assignment in assignments:
        if isinstance(assignment, dict) and assignment.get("assigned") and assignment.get("cluster_id"):
            cluster_id = str(assignment["cluster_id"])
            assignments_by_cluster_id[cluster_id] = assignments_by_cluster_id.get(cluster_id, 0) + 1

    ordered_clusters = sorted(
        clusters,
        key=lambda cluster: assignments_by_cluster_id.get(
            str(cluster.get("cluster_id")),
            len(cluster.get("frame_ids") or []),
        ),
        reverse=True,
    )
    strong_clusters = []
    for cluster in ordered_clusters:
        cluster_id = str(cluster.get("cluster_id"))
        assignment_count = assignments_by_cluster_id.get(cluster_id, 0)
        article_support = ((cluster.get("metadata") or {}) if isinstance(cluster.get("metadata"), dict) else {}).get("article_support", 0)
        if cluster.get("noise"):
            continue
        if max(assignment_count, article_support) < 3:
            continue
        strong_clusters.append(cluster)

    if not strong_clusters:
        st.info("Сильные нарративные кластеры не выделены. Текущий результат всё ещё слишком шумный.")
        return

    for cluster in strong_clusters[:20]:
        cluster_id = str(cluster.get("cluster_id"))
        label = labels_by_cluster_id.get(cluster_id)
        title = label.get("title") if isinstance(label, dict) and label.get("title") else cluster_id
        suffix = " [noise]" if cluster.get("noise") else ""
        metadata = cluster.get("metadata") or {}
        article_support = metadata.get("article_support", 0) if isinstance(metadata, dict) else 0
        source_support = metadata.get("source_support", 0) if isinstance(metadata, dict) else 0
        with st.expander(f"{title}{suffix}"):
            st.write(f"**Cluster ID:** {cluster_id}")
            st.write(f"**Тема:** {cluster.get('topic_id') or '—'}")
            st.write(f"**Фреймов в кластере:** {len(cluster.get('frame_ids') or [])}")
            st.write(f"**Поддержка по статьям:** {article_support or assignments_by_cluster_id.get(cluster_id, 0)}")
            st.write(f"**Поддержка по источникам:** {source_support or '—'}")
            if label is None:
                st.info("Лейбл кластера не сформирован.")
                continue
            st.write(f"**Описание:** {label.get('summary') or '—'}")
            st.write(f"**Canonical claim:** {label.get('canonical_claim') or '—'}")
            key_actors = label.get("key_actors") or []
            st.write(f"**Ключевые акторы:** {', '.join(key_actors) if key_actors else '—'}")
            causal_chain = label.get("causal_chain") or []
            st.write(f"**Причинная цепочка:** {' → '.join(causal_chain) if causal_chain else '—'}")
            st.write(f"**Доминирующая тональность:** {label.get('dominant_tone') or '—'}")
            st.write(f"**Контр-нарратив:** {label.get('counter_narrative') or '—'}")
            typical_formulations = label.get("typical_formulations") or []
            if typical_formulations:
                st.write("**Типичные формулировки:**")
                for item in typical_formulations[:5]:
                    st.markdown(f"- {item}")
            representative_examples = label.get("representative_examples") or []
            if representative_examples:
                st.write("**Representative examples:**")
                for item in representative_examples[:5]:
                    st.markdown(f"- {item}")


def _render_dynamics(snapshot: dict[str, object]) -> None:
    st.markdown("### Динамика")
    dynamics = list(snapshot.get("dynamics") or [])
    labels = list(snapshot.get("labels") or [])
    if not dynamics:
        st.info("Данные по динамике пока не выделены.")
        return

    labels_by_cluster_id = {
        str(label.get("cluster_id")): label
        for label in labels
        if isinstance(label, dict) and label.get("cluster_id") is not None
    }
    for series in dynamics[:12]:
        cluster_id = str(series.get("cluster_id"))
        label = labels_by_cluster_id.get(cluster_id)
        title = label.get("title") if isinstance(label, dict) and label.get("title") else cluster_id
        with st.expander(title):
            st.write(f"**Всего статей:** {series.get('total_articles', 0)}")
            st.write(
                f"**Рост:** {round(series['growth_rate'], 3) if series.get('growth_rate') is not None else '—'}"
            )
            st.write(
                f"**Устойчивость:** {round(series['stability_score'], 3) if series.get('stability_score') is not None else '—'}"
            )
            for point in series.get("points") or []:
                st.markdown(
                    f"- `{point.get('period_start', '—')}`: статей={point.get('article_count', 0)}, "
                    f"доля={round(point.get('share_of_corpus', 0.0), 3)}, "
                    f"diversity={round(point.get('source_diversity', 0.0), 3)}, "
                    f"burst={round(point['burst_score'], 3) if point.get('burst_score') is not None else '—'}"
                )


def render_narratives(services) -> None:
    st.subheader("Narrative Intelligence")
    st.caption(
        "Во вкладке показываются уже предрассчитанные narrative snapshots. "
        "Тяжёлый анализ запускается локально отдельной командой и сохраняется в БД."
    )

    source_options = get_source_options(services)
    selected_source_label = st.selectbox("Источник", list(source_options.keys()), key="narrative_source")
    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("Дата с", value=date(2026, 1, 1), key="narrative_date_from")
    with col2:
        date_to = st.date_input("Дата по", value=date.today(), key="narrative_date_to")

    date_from_value = f"{date_from.isoformat()}T00:00:00"
    date_to_value = f"{date_to.isoformat()}T23:59:59"
    source_domains = source_options[selected_source_label]
    source_domains_key = build_source_domains_key(source_domains)

    snapshot = services.narrative_analysis_repository.get_latest_payload(
        source_domains_key=source_domains_key,
        date_from=date_from_value,
        date_to=date_to_value,
    )
    run = services.narrative_analysis_repository.get_latest_run(
        source_domains_key=source_domains_key,
        date_from=date_from_value,
        date_to=date_to_value,
    )

    if snapshot is None or run is None:
        st.warning(
            "Для этого периода и источника готовый narrative snapshot не найден. "
            "Сначала выполни локальную materialization-команду, потом открывай интерфейс."
        )
        command = (
            f"C:\\Users\\79034\\anaconda3\\python.exe -m app materialize-narratives "
            f"--date-from {date_from_value} --date-to {date_to_value}"
        )
        if source_domains:
            command += f" --source-domains {','.join(source_domains)}"
        st.code(command)
        return

    st.caption(
        f"Snapshot run #{run.id} от {run.created_at}. "
        f"Источник: {source_domains_key}. Период: {run.date_from} — {run.date_to}."
    )

    st.markdown("### Сводка запуска")
    col1, col2, col3 = st.columns(3)
    col1.metric("Документы", run.documents_count)
    col2.metric("Темы", run.topics_count)
    col3.metric("Фреймы", run.frames_count)
    col4, col5, col6 = st.columns(3)
    col4.metric("Кластеры", run.clusters_count)
    col5.metric("Лейблы", run.labels_count)
    col6.metric("Назначения", run.assignments_count)

    frames = list(snapshot.get("frames") or [])
    no_clear_count = sum(
        1
        for frame in frames
        if isinstance(frame, dict) and frame.get("status") == "no_clear_narrative"
    )
    if no_clear_count:
        st.info(f"Статей без явного нарратива: {no_clear_count}")

    _render_topics(list(snapshot.get("topics") or []))
    _render_clusters(snapshot)
    _render_dynamics(snapshot)

    evaluation = snapshot.get("evaluation") if isinstance(snapshot, dict) else None
    if isinstance(evaluation, dict) and evaluation.get("notes"):
        with st.expander("Оценка качества"):
            for note in evaluation.get("notes") or []:
                st.markdown(f"- {note}")


def main() -> None:
    st.set_page_config(page_title="News Intelligence", layout="wide")
    st.title("News Intelligence")
    st.caption("Локальный интерфейс для анализа новостного корпуса: RAG и готовые narrative snapshots.")

    try:
        _connection, services, settings = get_connection_and_services()
    except Exception as exc:
        st.error(f"Не удалось инициализировать приложение: {exc}")
        return

    st.sidebar.markdown("### База данных")
    st.sidebar.code(str(settings.database_path))

    rag_tab, narrative_tab = st.tabs(["RAG", "Нарративы"])
    with rag_tab:
        render_rag(services)
    with narrative_tab:
        render_narratives(services)


if __name__ == "__main__":
    main()
