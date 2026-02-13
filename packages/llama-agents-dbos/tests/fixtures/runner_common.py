# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Shared utilities for subprocess test runners.

Importing this module adds all necessary package source directories to sys.path
as a side effect, so that runner scripts can import workflows, llama_agents, etc.
"""

from __future__ import annotations

import importlib
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
