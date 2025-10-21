"""Shared utilities for DBOS durability scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Coroutine

# Directory for storing state and database
BASE_DIR = Path(__file__).parent
DB_FILE = BASE_DIR / ".dbos_data.sqlite3"
LAST_RUN_FILE = BASE_DIR / ".last_run.json"

# Scenario registry
ScenarioFunc = Callable[[str, bool], Coroutine[None, None, None]]
SCENARIOS: dict[str, ScenarioFunc] = {}


def register_scenario(name: str) -> Callable[[ScenarioFunc], ScenarioFunc]:
    """Decorator to register a scenario function."""

    def decorator(func: ScenarioFunc) -> ScenarioFunc:
        SCENARIOS[name] = func
        return func

    return decorator
