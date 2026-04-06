"""Configuration and profile management for llamactl"""

import functools
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from ._migrations import run_migrations
from .schema import DEFAULT_ENVIRONMENT, Auth, DeviceOIDC, Environment


def _serialize_device_oidc(value: DeviceOIDC | None) -> str | None:
    if value is None:
        return None
    return value.model_dump_json()


def _deserialize_device_oidc(value: str | None) -> DeviceOIDC | None:
    if not value:
        return None
    return DeviceOIDC.model_validate_json(value)


def _to_auth(row: Any) -> Auth:
    return Auth(
        id=row[0],
        name=row[1],
        api_url=row[2],
        project_id=row[3],
        api_key=row[4],
        api_key_id=row[5],
        device_oidc=_deserialize_device_oidc(row[6]),
    )


def _to_environment(row: Any) -> Environment:
    return Environment(
        api_url=row[0],
        requires_auth=bool(row[1]),
        min_llamactl_version=row[2],
    )


class ConfigManager:
    """Manages profiles and configuration using SQLite"""

    def __init__(self, init_database: bool = True):
        self.config_dir = self._get_config_dir()
        self.db_path = self.config_dir / "profiles.db"
        self._ensure_config_dir()
        if init_database:
            self._init_database()

    def _get_config_dir(self) -> Path:
        """Get the configuration directory path based on OS.

        Honors LLAMACTL_CONFIG_DIR when set. This helps tests isolate state.
        """
        override = os.environ.get("LLAMACTL_CONFIG_DIR")
        if override:
            return Path(override).expanduser()
        if os.name == "nt":  # Windows
            config_dir = Path(os.environ.get("APPDATA", "~")) / "llamactl"
        else:  # Unix-like (Linux, macOS)
            config_dir = Path.home() / ".config" / "llamactl"
        return config_dir.expanduser()

    def _ensure_config_dir(self) -> None:
        """Create configuration directory if it doesn't exist"""
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def _init_database(self) -> None:
        """Initialize SQLite database and run migrations; then seed defaults."""

        with sqlite3.connect(self.db_path) as conn:
            # Apply ad-hoc SQL migrations based on PRAGMA user_version
            # Pass db_path to enable file-based locking across processes
            run_migrations(conn, self.db_path)

            conn.commit()

    def destroy_database(self) -> None:
        """Destroy the database"""
        self.db_path.unlink()
        self._init_database()

    #############################################
    ## Settings
    #############################################

    def set_settings_current_profile(self, name: str | None) -> None:
        """Set or clear the current active profile.

        If name is None, the setting is removed.
        """
        with sqlite3.connect(self.db_path) as conn:
            if name is None:
                conn.execute("DELETE FROM settings WHERE key = 'current_profile'")
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO settings (key, value) VALUES ('current_profile', ?)",
                    (name,),
                )
            conn.commit()

    def get_settings_current_profile_name(self) -> str | None:
        """Get the name of the current active profile"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM settings WHERE key = 'current_profile'"
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def set_settings_current_environment(self, api_url: str) -> None:
        """Set the current environment by URL.

        Requires the environment row to already exist (validated elsewhere, e.g. via
        a probe before creation). Raises ValueError if not found.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES ('current_environment_api_url', ?)",
                (api_url,),
            )
            conn.commit()

    def create_profile(
        self,
        name: str,
        api_url: str,
        project_id: str,
        api_key: str | None = None,
        api_key_id: str | None = None,
        device_oidc: DeviceOIDC | None = None,
    ) -> Auth:
        """Create a new auth profile"""
        if not project_id.strip():
            raise ValueError("Project ID is required")
        profile = Auth(
            id=str(uuid.uuid4()),
            name=name,
            api_url=api_url,
            project_id=project_id,
            api_key=api_key,
            api_key_id=api_key_id,
            device_oidc=device_oidc,
        )

        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute(
                    "INSERT INTO profiles (id, name, api_url, project_id, api_key, api_key_id, device_oidc) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        profile.id,
                        profile.name,
                        profile.api_url,
                        profile.project_id,
                        profile.api_key,
                        profile.api_key_id,
                        _serialize_device_oidc(profile.device_oidc),
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                raise ValueError(
                    f"Profile '{name}' already exists for environment '{api_url}'"
                )

        return profile

    def get_current_profile(self, env_url: str) -> Auth | None:
        """Get the current active profile"""
        current_name = self.get_settings_current_profile_name()
        if current_name:
            return self.get_profile(current_name, env_url)
        return None

    def get_current_environment(self) -> Environment:
        """Get the current active environment"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM settings WHERE key = 'current_environment_api_url'"
            )
            row = cursor.fetchone()
            api_url = row[0] if row else DEFAULT_ENVIRONMENT.api_url

            env_row = conn.execute(
                "SELECT api_url, requires_auth, min_llamactl_version FROM environments WHERE api_url = ?",
                (api_url,),
            ).fetchone()
            if env_row:
                return _to_environment(env_row)

        # Fallback: return an in-memory default without writing to DB
        if api_url == DEFAULT_ENVIRONMENT.api_url:
            return DEFAULT_ENVIRONMENT
        return Environment(
            api_url=api_url, requires_auth=False, min_llamactl_version=None
        )

    ##################################
    ## Profiles
    ##################################

    def get_profile(self, name: str, env_url: str) -> Auth | None:
        """Get a profile by name"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, name, api_url, project_id, api_key, api_key_id, device_oidc FROM profiles WHERE name = ? AND api_url = ?",
                (name, env_url),
            ).fetchone()
            if row:
                return _to_auth(row)
        return None

    def get_profile_by_id(self, id: str) -> Auth | None:
        """Get a profile by ID"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, name, api_url, project_id, api_key, api_key_id, device_oidc FROM profiles WHERE id = ?",
                (id,),
            ).fetchone()
            if row:
                return _to_auth(row)
        return None

    def get_profile_by_api_key(self, env_url: str, api_key: str) -> Auth | None:
        """Get a profile by api_key within an environment URL."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, name, api_url, project_id, api_key, api_key_id, device_oidc
                FROM profiles
                WHERE api_url = ? AND api_key = ?
                LIMIT 1
                """,
                (env_url, api_key),
            ).fetchone()
            if row:
                return _to_auth(row)
        return None

    def get_profile_by_device_user_id(self, env_url: str, user_id: str) -> Auth | None:
        """Get a profile by device OIDC user_id within an environment URL."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, name, api_url, project_id, api_key, api_key_id, device_oidc
                FROM profiles
                WHERE api_url = ? AND JSON_EXTRACT(device_oidc, '$.user_id') = ?
                LIMIT 1
                """,
                (env_url, user_id),
            ).fetchone()
            if row:
                return _to_auth(row)
        return None

    def list_profiles(self, env_url: str) -> list[Auth]:
        """List all profiles"""
        with sqlite3.connect(self.db_path) as conn:
            return [
                _to_auth(row)
                for row in conn.execute(
                    "SELECT id, name, api_url, project_id, api_key, api_key_id, device_oidc FROM profiles WHERE api_url = ? ORDER BY name",
                    (env_url,),
                ).fetchall()
            ]

    def delete_profile(self, name: str, env_url: str) -> bool:
        """Delete a profile by name. Returns True if deleted, False if not found."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM profiles WHERE name = ? AND api_url = ?", (name, env_url)
            )
            conn.commit()

            # If this was the active profile, clear it
            if self.get_settings_current_profile_name() == name:
                self.set_settings_current_profile(None)

            return cursor.rowcount > 0

    def set_project(self, profile_name: str, env_url: str, project_id: str) -> bool:
        """Set the project for a profile. Returns True if profile exists."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE profiles SET project_id = ? WHERE name = ? AND api_url = ?",
                (project_id, profile_name, env_url),
            )
            conn.commit()
            return cursor.rowcount > 0

    def update_profile(self, profile: Auth) -> None:
        """Update a profile"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE profiles SET name = ?, api_url = ?, project_id = ?, api_key = ?, api_key_id = ?, device_oidc = ? WHERE id = ?",
                (
                    profile.name,
                    profile.api_url,
                    profile.project_id,
                    profile.api_key,
                    profile.api_key_id,
                    _serialize_device_oidc(profile.device_oidc),
                    profile.id,
                ),
            )
            conn.commit()

    ##################################
    ## Environments
    ##################################
    def create_or_update_environment(
        self, api_url: str, requires_auth: bool, min_llamactl_version: str | None = None
    ) -> None:
        """Create or update an environment row."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO environments (api_url, requires_auth, min_llamactl_version) VALUES (?, ?, ?)",
                (api_url, 1 if requires_auth else 0, min_llamactl_version),
            )
            conn.commit()

    def get_environment(self, api_url: str) -> Environment | None:
        """Retrieve an environment by URL."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT api_url, requires_auth, min_llamactl_version FROM environments WHERE api_url = ?",
                (api_url,),
            ).fetchone()
            if row:
                return _to_environment(row)
        return None

    def list_environments(self) -> list[Environment]:
        """List all environments."""
        with sqlite3.connect(self.db_path) as conn:
            envs = [
                _to_environment(row)
                for row in conn.execute(
                    "SELECT api_url, requires_auth, min_llamactl_version FROM environments ORDER BY api_url"
                ).fetchall()
            ]
            if not envs:
                envs = [DEFAULT_ENVIRONMENT]
            return envs

    def delete_environment(self, api_url: str) -> bool:
        """Delete an environment and all associated profiles.

        Returns True if the environment existed and was deleted, False otherwise.
        If the deleted environment was current, switch current to the default URL.
        """
        with sqlite3.connect(self.db_path) as conn:
            # Check existence
            exists_cursor = conn.execute(
                "SELECT 1 FROM environments WHERE api_url = ?",
                (api_url,),
            )
            if exists_cursor.fetchone() is None:
                return False

            # Delete profiles tied to this environment
            conn.execute("DELETE FROM profiles WHERE api_url = ?", (api_url,))

            # Delete environment row
            conn.execute("DELETE FROM environments WHERE api_url = ?", (api_url,))

            # If current environment is this one, reset to default
            setting_cursor = conn.execute(
                "SELECT value FROM settings WHERE key = 'current_environment_api_url'"
            )
            row = setting_cursor.fetchone()
            if row and row[0] == api_url:
                conn.execute(
                    "INSERT OR REPLACE INTO settings (key, value) VALUES ('current_environment_api_url', ?)",
                    (DEFAULT_ENVIRONMENT.api_url,),
                )

            conn.commit()
            return True


# Global config manager instance
@functools.cache
def config_manager() -> ConfigManager:
    return ConfigManager()
