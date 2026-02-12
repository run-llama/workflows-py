# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Shared utilities for subprocess test runners.

Importing this module adds all necessary package source directories to sys.path
as a side effect, so that runner scripts can import workflows, llama_agents, etc.
"""

from __future__ import annotations

import importlib
import sqlite3
import sys
from pathlib import Path
from types import ModuleType

# Compute package source directories relative to this file
TESTS_DIR = Path(__file__).parent.parent
DBOS_PACKAGE_DIR = TESTS_DIR.parent
_SYS_PATHS = [
    str(DBOS_PACKAGE_DIR),
    str(DBOS_PACKAGE_DIR / "src"),
    str(DBOS_PACKAGE_DIR.parent / "llama-index-workflows" / "src"),
    str(DBOS_PACKAGE_DIR.parent / "llama-agents-server" / "src"),
    str(DBOS_PACKAGE_DIR.parent / "llama-agents-client" / "src"),
    str(DBOS_PACKAGE_DIR.parent / "llama-index-instrumentation" / "src"),
]

for _p in _SYS_PATHS:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dbos import DBOS, DBOSConfig  # noqa: E402
from llama_agents.dbos import DBOSRuntime  # noqa: E402
from workflows.events import Event  # noqa: E402
from workflows.workflow import Workflow  # noqa: E402


def import_workflow(path: str) -> tuple[type[Workflow], ModuleType]:
    """Import a workflow class from a module path like 'module.path:ClassName'."""
    if ":" not in path:
        raise ValueError(f"Invalid workflow path format: {path}")
    module_path, class_name = path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    workflow_class = getattr(module, class_name)
    if not (isinstance(workflow_class, type) and issubclass(workflow_class, Workflow)):
        raise TypeError(f"{class_name} is not a Workflow subclass")
    return workflow_class, module


def get_event_class_by_name(module: ModuleType, name: str) -> type[Event] | None:
    """Find an event class in a module by its name."""
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, Event) and attr.__name__ == name:
            return attr
    return None


def setup_dbos(db_url: str, app_name: str = "test-workflow") -> DBOSRuntime:
    """Set up DBOS with the given database URL and return a DBOSRuntime."""
    config: DBOSConfig = {
        "name": app_name,
        "system_database_url": db_url,
        "run_admin_server": False,
        "notification_listener_polling_interval_sec": 0.01,
    }
    DBOS(config=config)
    return DBOSRuntime(polling_interval_sec=0.01)


def dump_dbos_operations(db_url: str, run_id: str) -> None:
    """Dump DBOS operations from the database for debugging.

    Extracts the SQLite file path from the db_url, connects to the database,
    and prints all tables with their row counts, plus full contents of any
    operation-related tables.

    Args:
        db_url: SQLite database URL (format: sqlite+pysqlite:///path/to/db?...)
        run_id: Workflow run ID to filter for
    """
    try:
        # Extract file path from URL (e.g., "sqlite+pysqlite:///path/to/db?...")
        # Remove the scheme prefix
        if ":///" in db_url:
            path_part = db_url.split(":///", 1)[1]
            # Remove any query parameters
            db_path = path_part.split("?", 1)[0]
        else:
            print(f"Could not parse db_url: {db_url}")
            return

        # Check if file exists
        if not Path(db_path).exists():
            print(f"Database file does not exist: {db_path}")
            return

        # Connect to SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        print("\n" + "=" * 80)
        print(f"DBOS DATABASE DUMP - run_id: {run_id}")
        print(f"Database: {db_path}")
        print("=" * 80)

        # Get all table names
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        # Print table list with row counts
        print("\nTABLES:")
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  {table}: {count} rows")

        # Dump contents of relevant tables
        operation_tables = [
            t
            for t in tables
            if "operation" in t.lower()
            or "step" in t.lower()
            or "journal" in t.lower()
            or "stream" in t.lower()
        ]

        for table in operation_tables:
            print(f"\n{'-' * 80}")
            print(f"TABLE: {table}")
            print(f"{'-' * 80}")

            # Get column names
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            print(f"Columns: {', '.join(columns)}")

            # Fetch all rows
            cursor.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()

            if not rows:
                print("  (no rows)")
            else:
                # Print each row
                for i, row in enumerate(rows, 1):
                    print(f"\nRow {i}:")
                    for col, val in zip(columns, row):
                        # Truncate long values for readability
                        val_str = str(val)
                        if len(val_str) > 200:
                            val_str = val_str[:200] + "..."
                        print(f"  {col}: {val_str}")

        print("\n" + "=" * 80)
        print("END DBOS DATABASE DUMP")
        print("=" * 80 + "\n")

        conn.close()

    except Exception as e:
        print(f"Error dumping DBOS operations: {type(e).__name__}: {e}")
