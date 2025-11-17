from __future__ import annotations

try:
    from importlib.resources.abc import Traversable  # type: ignore
except ImportError:  # pre 3.11
    from importlib.abc import Traversable  # type: ignore
import logging
import re
import sqlite3
from importlib import import_module, resources

logger = logging.getLogger(__name__)


_MIGRATIONS_PKG = "workflows.server.sqlite.migrations"
_USER_VERSION_PATTERN = re.compile(r"pragma\s+user_version\s*=\s*(\d+)", re.IGNORECASE)


def _iter_migration_files() -> list[Traversable]:
    """Yield packaged SQL migration files in lexicographic order."""
    pkg = import_module(_MIGRATIONS_PKG)
    root = resources.files(pkg)
    files = (p for p in root.iterdir() if p.name.endswith(".sql"))
    return sorted(files, key=lambda p: p.name)  # type: ignore


def _parse_target_version(sql_text: str) -> int | None:
    """Return target schema version declared in the first PRAGMA line, if any."""
    first_line = sql_text.splitlines()[0] if sql_text else ""
    match = _USER_VERSION_PATTERN.search(first_line)
    return int(match.group(1)) if match else None


def run_migrations(conn: sqlite3.Connection) -> None:
    """Apply pending migrations found under the migrations package.

    Each migration file should start with a `PRAGMA user_version=N;` line.
    Files are applied in lexicographic order and only when N > current_version.
    """
    cur = conn.cursor()
    current_version_row = cur.execute("PRAGMA user_version").fetchone()
    current_version = int(current_version_row[0]) if current_version_row else 0

    for path in _iter_migration_files():
        sql_text = path.read_text()
        target_version = _parse_target_version(sql_text) or 0
        if target_version <= current_version:
            continue

        try:
            logger.debug(
                "Applying migration %s → target version %s", path.name, target_version
            )
            cur.executescript("BEGIN;\n" + sql_text)
        except Exception as exc:  # noqa: BLE001 – we surface the exact error
            logger.error("Failed migration %s: %s", path.name, exc)
            cur.execute("ROLLBACK")
            raise
        else:
            cur.execute("COMMIT")
            current_version = target_version
