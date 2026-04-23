# NarrativeNews

Foundation layer for a Russian-language news analysis system.

## What is included

- Python project scaffold
- SQLite schema and database initialization
- Dataclass-based domain models
- Repository layer for core entities
- Ingestion foundation with source-specific parsers for Lenta.ru and ria.ru
- Minimal CLI entrypoint
- Example usage script
- Basic test coverage

## Quick start

```bash
python -m app init-db
python -m examples.basic_usage
python -m unittest discover -s tests -v
```

## Ingestion example

```bash
python -m app ingest-source lenta --limit 5
python -m app ingest-source ria --limit 5
```
