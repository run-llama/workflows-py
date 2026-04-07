from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from llama_agents.cli.config._config import ConfigManager


def test_delete_environment_cascades_and_resets_current() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        cfg = ConfigManager()
        cfg.config_dir = Path(temp_dir)
        cfg.db_path = cfg.config_dir / "profiles.db"
        cfg._ensure_config_dir()
        cfg._init_database()

        url = "https://env.del.local"
        cfg.create_or_update_environment(url, requires_auth=False)
        cfg.create_profile("a", url, "p")
        cfg.set_settings_current_environment(url)
        assert cfg.get_current_environment().api_url == url

        deleted = cfg.delete_environment(url)
        assert deleted is True
        # Current env should reset to default, profiles removed
        current = cfg.get_current_environment()
        assert current.api_url != url
        assert cfg.list_profiles(url) == []

        # Deleting again should return False
        assert cfg.delete_environment(url) is False


def test_get_current_environment_fallback_when_missing_row() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        cfg = ConfigManager()
        cfg.config_dir = Path(temp_dir)
        cfg.db_path = cfg.config_dir / "profiles.db"
        cfg._ensure_config_dir()
        cfg._init_database()

        # Manually set a current env URL without adding an env row
        missing_url = "https://missing.local"
        with sqlite3.connect(cfg.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES ('current_environment_api_url', ?)",
                (missing_url,),
            )
            conn.commit()

        env = cfg.get_current_environment()
        assert env.api_url == missing_url
        assert env.requires_auth is False
        assert env.min_llamactl_version is None
