from __future__ import annotations

import sqlite3


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    domain TEXT NOT NULL UNIQUE,
    base_url TEXT NOT NULL,
    source_type TEXT NOT NULL,
    language TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    subtitle TEXT,
    body_text TEXT NOT NULL,
    published_at TEXT NOT NULL,
    author TEXT,
    category TEXT,
    language TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    word_count INTEGER NOT NULL DEFAULT 0 CHECK (word_count >= 0),
    is_canonical INTEGER NOT NULL DEFAULT 1 CHECK (is_canonical IN (0, 1)),
    duplicate_group_id TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS article_duplicates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    duplicate_group_id TEXT NOT NULL,
    article_id INTEGER NOT NULL,
    duplicate_type TEXT NOT NULL,
    is_primary INTEGER NOT NULL DEFAULT 0 CHECK (is_primary IN (0, 1)),
    similarity_score REAL CHECK (similarity_score IS NULL OR (similarity_score >= 0 AND similarity_score <= 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS article_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    char_start INTEGER NOT NULL CHECK (char_start >= 0),
    char_end INTEGER NOT NULL CHECK (char_end >= char_start),
    token_count INTEGER CHECK (token_count IS NULL OR token_count >= 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE,
    UNIQUE (article_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL,
    claim_text TEXT NOT NULL,
    normalized_claim_text TEXT,
    claim_type TEXT NOT NULL,
    extraction_confidence REAL CHECK (
        extraction_confidence IS NULL OR (extraction_confidence >= 0 AND extraction_confidence <= 1)
    ),
    classification_confidence REAL CHECK (
        classification_confidence IS NULL OR (classification_confidence >= 0 AND classification_confidence <= 1)
    ),
    source_sentence TEXT,
    source_paragraph_index INTEGER CHECK (source_paragraph_index IS NULL OR source_paragraph_index >= 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS narrative_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_text TEXT NOT NULL,
    date_from TEXT NOT NULL,
    date_to TEXT NOT NULL,
    run_status TEXT NOT NULL,
    articles_selected_count INTEGER NOT NULL DEFAULT 0 CHECK (articles_selected_count >= 0),
    claims_selected_count INTEGER NOT NULL DEFAULT 0 CHECK (claims_selected_count >= 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS claim_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    claim_type TEXT NOT NULL,
    cluster_label TEXT NOT NULL,
    cluster_summary TEXT,
    cluster_score REAL CHECK (cluster_score IS NULL OR (cluster_score >= 0 AND cluster_score <= 1)),
    claim_count INTEGER NOT NULL DEFAULT 0 CHECK (claim_count >= 0),
    article_count INTEGER NOT NULL DEFAULT 0 CHECK (article_count >= 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES narrative_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS claim_cluster_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER NOT NULL,
    claim_id INTEGER NOT NULL,
    membership_score REAL CHECK (membership_score IS NULL OR (membership_score >= 0 AND membership_score <= 1)),
    is_representative INTEGER NOT NULL DEFAULT 0 CHECK (is_representative IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (cluster_id) REFERENCES claim_clusters(id) ON DELETE CASCADE,
    FOREIGN KEY (claim_id) REFERENCES claims(id) ON DELETE CASCADE,
    UNIQUE (cluster_id, claim_id)
);

CREATE TABLE IF NOT EXISTS narrative_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    narrative_type TEXT NOT NULL,
    title TEXT NOT NULL,
    formulation TEXT NOT NULL,
    explanation TEXT,
    strength_score REAL CHECK (strength_score IS NULL OR (strength_score >= 0 AND strength_score <= 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES narrative_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS narrative_result_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    narrative_result_id INTEGER NOT NULL,
    article_id INTEGER NOT NULL,
    rank INTEGER NOT NULL CHECK (rank >= 1),
    selection_reason TEXT,
    FOREIGN KEY (narrative_result_id) REFERENCES narrative_results(id) ON DELETE CASCADE,
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE,
    UNIQUE (narrative_result_id, article_id)
);

CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at);
CREATE INDEX IF NOT EXISTS idx_articles_source_id ON articles(source_id);
CREATE INDEX IF NOT EXISTS idx_articles_is_canonical ON articles(is_canonical);
CREATE INDEX IF NOT EXISTS idx_articles_content_hash ON articles(content_hash);
CREATE INDEX IF NOT EXISTS idx_articles_duplicate_group_id ON articles(duplicate_group_id);
CREATE INDEX IF NOT EXISTS idx_claims_article_id ON claims(article_id);
CREATE INDEX IF NOT EXISTS idx_claims_claim_type ON claims(claim_type);
CREATE INDEX IF NOT EXISTS idx_narrative_runs_topic_text ON narrative_runs(topic_text);
CREATE INDEX IF NOT EXISTS idx_claim_clusters_run_id ON claim_clusters(run_id);
CREATE INDEX IF NOT EXISTS idx_claim_clusters_claim_type ON claim_clusters(claim_type);
"""


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA_SQL)
    connection.commit()
