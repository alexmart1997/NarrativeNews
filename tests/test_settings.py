from __future__ import annotations

import os
from pathlib import Path
import shutil
import unittest
import uuid
from unittest.mock import patch

from app.config.settings import _resolve_database_path


class SettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path("tests") / ".tmp" / f"settings-{uuid.uuid4().hex}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.default_path = self.temp_dir / "narrative_news.db"
        self.deploy_path = self.temp_dir / "narrative_news_deploy.db"

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_env_database_path_has_priority(self) -> None:
        with patch.dict(os.environ, {"NARRATIVE_NEWS_DB_PATH": str(self.temp_dir / "custom.db")}, clear=False):
            with patch("app.config.settings.DEFAULT_DB_PATH", self.default_path), patch(
                "app.config.settings.DEPLOY_DB_PATH",
                self.deploy_path,
            ):
                self.assertEqual(_resolve_database_path(), self.temp_dir / "custom.db")

    def test_uses_deploy_snapshot_when_default_is_missing(self) -> None:
        self.deploy_path.write_bytes(b"x" * 2_000_000)
        with patch.dict(os.environ, {}, clear=True):
            with patch("app.config.settings.DEFAULT_DB_PATH", self.default_path), patch(
                "app.config.settings.DEPLOY_DB_PATH",
                self.deploy_path,
            ):
                self.assertEqual(_resolve_database_path(), self.deploy_path)

    def test_uses_deploy_snapshot_when_default_is_tiny(self) -> None:
        self.default_path.write_bytes(b"x" * 128)
        self.deploy_path.write_bytes(b"x" * 2_000_000)
        with patch.dict(os.environ, {}, clear=True):
            with patch("app.config.settings.DEFAULT_DB_PATH", self.default_path), patch(
                "app.config.settings.DEPLOY_DB_PATH",
                self.deploy_path,
            ):
                self.assertEqual(_resolve_database_path(), self.deploy_path)

    def test_uses_default_database_when_it_is_real(self) -> None:
        self.default_path.write_bytes(b"x" * 2_000_000)
        self.deploy_path.write_bytes(b"x" * 2_000_000)
        with patch.dict(os.environ, {}, clear=True):
            with patch("app.config.settings.DEFAULT_DB_PATH", self.default_path), patch(
                "app.config.settings.DEPLOY_DB_PATH",
                self.deploy_path,
            ):
                self.assertEqual(_resolve_database_path(), self.default_path)


if __name__ == "__main__":
    unittest.main()
