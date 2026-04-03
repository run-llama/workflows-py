"""Tests for config.py - Database operations and profile management"""

import sqlite3
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from llama_agents.cli.config._config import ConfigManager


@pytest.fixture
def temp_config() -> Generator[ConfigManager, None, None]:
    """Create a temporary config manager for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_manager = ConfigManager()
        # Override the config directory to use temp directory
        config_manager.config_dir = Path(temp_dir)
        config_manager.db_path = config_manager.config_dir / "profiles.db"
        config_manager._ensure_config_dir()
        config_manager._init_database()
        yield config_manager


def test_create_profile(temp_config: ConfigManager) -> None:
    """Test profile creation with CRUD operations"""
    env_url = "http://localhost:8011"
    # Create a profile
    profile = temp_config.create_profile("test", env_url, "test-project")

    assert profile.name == "test"
    assert profile.api_url == env_url
    assert profile.project_id == "test-project"

    # Retrieve the profile
    retrieved = temp_config.get_profile("test", env_url)
    assert retrieved is not None
    assert retrieved.name == "test"
    assert retrieved.api_url == env_url
    assert retrieved.project_id == "test-project"

    # Test duplicate creation fails within the same environment
    with pytest.raises(ValueError, match="Profile 'test' already exists"):
        temp_config.create_profile("test", env_url, "other-project")

    # Creating a profile with the same name in a different environment should succeed
    other_env_url = "http://other:8011"
    other_profile = temp_config.create_profile("test", other_env_url, "other")
    assert other_profile.api_url == other_env_url

    # Test profile without project - this should now fail since project_id is required
    with pytest.raises(ValueError):
        temp_config.create_profile("minimal", "http://localhost:8012", "")

    # List profiles - should only have 1 now since the second creation failed
    profiles = temp_config.list_profiles(env_url)
    assert len(profiles) == 1
    assert profiles[0].name == "test"

    # Delete profile
    assert temp_config.delete_profile("test", env_url) is True
    assert temp_config.get_profile("test", env_url) is None
    assert temp_config.delete_profile("nonexistent", env_url) is False


def test_profile_migration(temp_config: ConfigManager) -> None:
    """Test migrating from 0001 schema to 0002 (adds id, api_key_id, device_oidc)."""
    # Simulate pre-0002 database with 0001 schema
    with sqlite3.connect(temp_config.db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS profiles")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                name TEXT NOT NULL,
                api_url TEXT NOT NULL,
                project_id TEXT NOT NULL,
                api_key TEXT,
                PRIMARY KEY (name, api_url)
            )
            """
        )
        # Insert sample row
        conn.execute(
            "INSERT INTO profiles (name, api_url, project_id, api_key) VALUES (?, ?, ?, ?)",
            ("with-project", "http://legacy:8011", "legacy-project", None),
        )
        # Mark DB as version 1
        conn.execute("PRAGMA user_version=1")
        conn.commit()

    # Trigger migrations (should apply 0002)
    migrated_config = ConfigManager()
    migrated_config.config_dir = temp_config.config_dir
    migrated_config.db_path = temp_config.db_path
    migrated_config._init_database()

    # Row preserved and retrievable
    profile = migrated_config.get_profile("with-project", "http://legacy:8011")
    assert profile is not None
    assert profile.project_id == "legacy-project"

    # Schema is updated with new columns
    with sqlite3.connect(migrated_config.db_path) as conn:
        cursor = conn.execute("PRAGMA table_info(profiles)")
        columns = [row[1] for row in cursor.fetchall()]
        assert "project_id" in columns
        assert "api_key" in columns
        assert "id" in columns
        assert "api_key_id" in columns
        assert "device_oidc" in columns


def test_project_management(temp_config: ConfigManager) -> None:
    """Test setting and getting projects"""
    env_url = "http://localhost:8011"
    # Create a profile
    temp_config.create_profile("test", env_url, "initial-project")

    # Test initial project
    prof = temp_config.get_profile("test", env_url)
    assert prof is not None
    assert prof.project_id == "initial-project"

    # Set new project
    assert temp_config.set_project("test", env_url, "new-project") is True
    prof = temp_config.get_profile("test", env_url)
    assert prof is not None
    assert prof.project_id == "new-project"

    # Projects are now required, so we can't set to None
    # Instead test setting to a different project
    assert temp_config.set_project("test", env_url, "another-project") is True
    prof = temp_config.get_profile("test", env_url)
    assert prof is not None
    assert prof.project_id == "another-project"

    # Test with nonexistent profile
    assert temp_config.set_project("nonexistent", env_url, "project") is False
    assert temp_config.get_profile("nonexistent", env_url) is None

    # Test current profile integration
    temp_config.set_settings_current_profile("test")
    current = temp_config.get_current_profile(env_url)
    assert current is not None
    assert current.name == "test"
    assert current.project_id == "another-project"

    # Set project and verify it's reflected in current profile
    temp_config.set_project("test", env_url, "final-project")
    current = temp_config.get_current_profile(env_url)
    assert current is not None
    assert current.project_id == "final-project"

    # Test deleting profile clears current setting
    temp_config.delete_profile("test", env_url)
    assert temp_config.get_current_profile(env_url) is None


def test_environments_table_and_default_current_environment(
    temp_config: ConfigManager,
) -> None:
    """Fresh DB should have environments table and default current environment set."""
    # Verify settings contains current_environment_api_url
    with sqlite3.connect(temp_config.db_path) as conn:
        cursor = conn.execute(
            "SELECT value FROM settings WHERE key = 'current_environment_api_url'"
        )
        row = cursor.fetchone()
        assert row is not None
        current_env_url = row[0]

        # Verify environments table exists and has the current env row
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='environments'"
        )
        assert cursor.fetchone() is not None

        env_cursor = conn.execute(
            "SELECT api_url, requires_auth FROM environments WHERE api_url = ?",
            (current_env_url,),
        )
        env_row = env_cursor.fetchone()
        assert env_row is not None
        assert env_row[0] == current_env_url
        # Default seed uses requires_auth = 0
        assert env_row[1] in (0, 1)


def test_environment_seed_from_profiles_migration(temp_config: ConfigManager) -> None:
    """Existing DB with 0001-era profiles should seed environments when migrating from version 0."""
    # Prepare DB with distinct profiles and simulate pre-0001 state (no environments, user_version=0)
    with sqlite3.connect(temp_config.db_path) as conn:
        # Start from a clean slate: drop tables and set version 0
        conn.execute("DROP TABLE IF EXISTS profiles")
        conn.execute("DROP TABLE IF EXISTS environments")
        conn.execute("DROP TABLE IF EXISTS settings")
        conn.execute("PRAGMA user_version=0")

        # Create a minimal 0001-style profiles table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                name TEXT NOT NULL,
                api_url TEXT NOT NULL,
                project_id TEXT NOT NULL,
                api_key TEXT,
                PRIMARY KEY (name, api_url)
            )
            """
        )
        conn.execute(
            "INSERT INTO profiles (name, api_url, project_id, api_key) VALUES (?, ?, ?, ?)",
            ("p1", "http://env-a:8000", "proj-a", None),
        )
        conn.execute(
            "INSERT INTO profiles (name, api_url, project_id, api_key) VALUES (?, ?, ?, ?)",
            ("p2", "http://env-b:8000", "proj-b", None),
        )
        conn.commit()

    # Re-run initialization to trigger 0001 creation and seeding then 0002 migration
    temp_config._init_database()

    with sqlite3.connect(temp_config.db_path) as conn:
        # Environments should include both profile envs plus ensure default exists
        envs = {
            row[0]
            for row in conn.execute("SELECT api_url FROM environments").fetchall()
        }
        assert "http://env-a:8000" in envs
        assert "http://env-b:8000" in envs
        cur = conn.execute(
            "SELECT value FROM settings WHERE key = 'current_environment_api_url'"
        ).fetchone()
        assert cur is not None
        # setting must be a string; it may or may not be one of the above
        assert isinstance(cur[0], str)


def test_environment_methods_and_current_behavior(temp_config: ConfigManager) -> None:
    """Validate environment CRUD and get_current_profile preference."""
    # Add a new environment and set requires_auth
    temp_config.create_or_update_environment(
        "http://custom-env:9000", True, min_llamactl_version="0.3.0a13"
    )
    env = temp_config.get_environment("http://custom-env:9000")
    assert env is not None
    assert env.api_url == "http://custom-env:9000"
    assert env.requires_auth is True
    assert env.min_llamactl_version == "0.3.0a13"

    # List environments includes the new one
    envs = temp_config.list_environments()
    assert any(e.api_url == "http://custom-env:9000" for e in envs)

    # Set current environment to a URL with a single profile and verify listing
    env_only_url = "http://only-here:7777"
    temp_config.create_profile("only-here", env_only_url, "proj-one")
    # Ensure the environment exists first (simulating validated add)
    temp_config.create_or_update_environment(env_only_url, False)
    temp_config.set_settings_current_environment(env_only_url)
    env_profiles = temp_config.list_profiles(env_only_url)
    assert len(env_profiles) == 1
    assert env_profiles[0].name == "only-here"
    # Set as current and verify get_current_profile returns it for this env
    temp_config.set_settings_current_profile("only-here")
    preferred = temp_config.get_current_profile(env_only_url)
    assert preferred is not None
    assert preferred.name == "only-here"
