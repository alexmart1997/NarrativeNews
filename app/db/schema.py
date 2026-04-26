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

CREATE VIRTUAL TABLE IF NOT EXISTS article_chunks_fts USING fts5(
    chunk_text,
    article_title,
    content='',
    tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS article_chunk_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chunk_id) REFERENCES article_chunks(id) ON DELETE CASCADE,
    UNIQUE (chunk_id, model_name)
);

CREATE TABLE IF NOT EXISTS narrative_analysis_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_domains_key TEXT NOT NULL,
    date_from TEXT NOT NULL,
    date_to TEXT NOT NULL,
    status TEXT NOT NULL,
    documents_count INTEGER NOT NULL DEFAULT 0 CHECK (documents_count >= 0),
    topics_count INTEGER NOT NULL DEFAULT 0 CHECK (topics_count >= 0),
    frames_count INTEGER NOT NULL DEFAULT 0 CHECK (frames_count >= 0),
    clusters_count INTEGER NOT NULL DEFAULT 0 CHECK (clusters_count >= 0),
    labels_count INTEGER NOT NULL DEFAULT 0 CHECK (labels_count >= 0),
    assignments_count INTEGER NOT NULL DEFAULT 0 CHECK (assignments_count >= 0),
    dynamics_count INTEGER NOT NULL DEFAULT 0 CHECK (dynamics_count >= 0),
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at);
CREATE INDEX IF NOT EXISTS idx_articles_source_id ON articles(source_id);
CREATE INDEX IF NOT EXISTS idx_articles_is_canonical ON articles(is_canonical);
CREATE INDEX IF NOT EXISTS idx_articles_content_hash ON articles(content_hash);
CREATE INDEX IF NOT EXISTS idx_articles_duplicate_group_id ON articles(duplicate_group_id);
CREATE INDEX IF NOT EXISTS idx_article_chunks_article_id ON article_chunks(article_id);
CREATE INDEX IF NOT EXISTS idx_article_chunk_embeddings_chunk_id ON article_chunk_embeddings(chunk_id);
CREATE INDEX IF NOT EXISTS idx_article_chunk_embeddings_model_name ON article_chunk_embeddings(model_name);
CREATE INDEX IF NOT EXISTS idx_narrative_analysis_runs_lookup
ON narrative_analysis_runs(source_domains_key, date_from, date_to, created_at DESC);

CREATE TRIGGER IF NOT EXISTS trg_article_chunks_fts_insert
AFTER INSERT ON article_chunks
BEGIN
    INSERT INTO article_chunks_fts(rowid, chunk_text, article_title)
    VALUES (
        NEW.id,
        NEW.chunk_text,
        COALESCE((SELECT title FROM articles WHERE id = NEW.article_id), '')
    );
END;

CREATE TRIGGER IF NOT EXISTS trg_article_chunks_fts_delete
AFTER DELETE ON article_chunks
BEGIN
    DELETE FROM article_chunks_fts WHERE rowid = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_article_chunks_fts_update
AFTER UPDATE OF chunk_text, article_id ON article_chunks
BEGIN
    DELETE FROM article_chunks_fts WHERE rowid = OLD.id;
    INSERT INTO article_chunks_fts(rowid, chunk_text, article_title)
    VALUES (
        NEW.id,
        NEW.chunk_text,
        COALESCE((SELECT title FROM articles WHERE id = NEW.article_id), '')
    );
END;
"""


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA_SQL)
    connection.execute(
        """
        INSERT INTO article_chunks_fts(rowid, chunk_text, article_title)
        SELECT
            ac.id,
            ac.chunk_text,
            a.title
        FROM article_chunks ac
        INNER JOIN articles a ON a.id = ac.article_id
        WHERE NOT EXISTS (
            SELECT 1
            FROM article_chunks_fts fts
            WHERE fts.rowid = ac.id
        )
        """
    )
    connection.commit()
