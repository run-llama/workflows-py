# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Subprocess runner for workflow tests with DBOS isolation.

This module provides a CLI runner for executing workflows in isolated
subprocesses, supporting interrupt/resume testing and human-in-the-loop
response simulation.

Usage:
    python /path/to/packages/llama-agents-dbos/tests/fixtures/runner.py \
        --workflow "tests.fixtures.workflows.hitl:TestWorkflow" \
        --db-url "sqlite+pysqlite:///path/to/db" \
        --run-id "test-001" \
        --config '{"interrupt_on": "AskInputEvent"}'

Config modes:
    - interrupt_on: Interrupt when event type is seen (uses os._exit(0))
      - String form: "EventName" - interrupt on any instance of EventName
      - Dict form: {"event": "EventName", "condition": {"field": value}}
                   - interrupt only when type matches AND all condition fields match
    - respond: Respond to InputRequiredEvent subtypes with specified events
    - run-to-completion: Empty config or omit both fields
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

# Add package source directories to sys.path for imports
# Runner is at: packages/llama-agents-dbos/tests/fixtures/runner.py
# We need to add:
#   - packages/llama-agents-dbos/src for llama_agents.dbos
#   - packages/llama-index-workflows/src for workflows.*
#   - packages/llama-agents-dbos (parent of tests/) so tests.fixtures.workflows.* can be imported
TESTS_DIR = Path(__file__).parent.parent
DBOS_PACKAGE_DIR = TESTS_DIR.parent
DBOS_PACKAGE_SRC_PATH = str(DBOS_PACKAGE_DIR / "src")
WORKFLOWS_PACKAGE_SRC_PATH = str(
    DBOS_PACKAGE_DIR.parent / "llama-index-workflows" / "src"
)

# Insert at front of path so these packages take precedence
# Add the parent of tests/ so "import tests.fixtures.workflows..." works
sys.path.insert(0, str(DBOS_PACKAGE_DIR))
sys.path.insert(0, DBOS_PACKAGE_SRC_PATH)
sys.path.insert(0, WORKFLOWS_PACKAGE_SRC_PATH)

from dbos import DBOS, DBOSConfig  # noqa: E402
from llama_agents.dbos import DBOSRuntime  # noqa: E402
from workflows.context import Context  # noqa: E402
from workflows.events import Event, InputRequiredEvent, StartEvent  # noqa: E402
from workflows.workflow import Workflow  # noqa: E402


def import_workflow(path: str) -> tuple[type[Workflow], ModuleType]:
    """Import a workflow class from a module path.

    Args:
        path: Module path with class name, e.g., "tests.fixtures.workflows.hitl:TestWorkflow"

    Returns:
        Tuple of (workflow_class, module) for accessing classes defined in the module.

    Raises:
        ValueError: If path format is invalid.
        ImportError: If module cannot be imported.
        AttributeError: If class not found in module.
    """
    if ":" not in path:
        raise ValueError(
            f"Invalid workflow path format: {path}. Expected 'module.path:ClassName'"
        )
    module_path, class_name = path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    workflow_class = getattr(module, class_name)
    if not (isinstance(workflow_class, type) and issubclass(workflow_class, Workflow)):
        raise TypeError(f"{class_name} is not a Workflow subclass")
    return workflow_class, module


def get_event_class_by_name(module: ModuleType, name: str) -> type[Event] | None:
    """Find an event class in a module by its name.

    Searches through all attributes of the module to find an Event subclass
    with a matching class name.

    Args:
        module: The module to search in.
        name: The class name to find.

    Returns:
        The event class if found, None otherwise.
    """
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, Event) and attr.__name__ == name:
            return attr
    return None


def parse_config(config_json: str | None) -> dict[str, Any]:
    """Parse the JSON config string.

    Args:
        config_json: JSON string with configuration, or None.

    Returns:
        Parsed config dict, or empty dict if None.
    """
    if not config_json:
        return {}
    return json.loads(config_json)


def setup_dbos(db_url: str, app_name: str = "test-workflow") -> DBOSRuntime:
    """Set up DBOS with the given database URL.

    Args:
        db_url: SQLite database URL.
        app_name: Application name for DBOS config.

    Returns:
        Configured DBOSRuntime instance.
    """
    config: DBOSConfig = {
        "name": app_name,
        "system_database_url": db_url,
        "run_admin_server": False,
        "notification_listener_polling_interval_sec": 0.01,
    }
    DBOS(config=config)
    return DBOSRuntime(polling_interval_sec=0.01)


async def run_workflow(
    workflow_path: str,
    db_url: str,
    run_id: str,
    config: dict[str, Any],
) -> None:
    """Run the workflow with the specified configuration.

    Args:
        workflow_path: Module path with class name.
        db_url: SQLite database URL.
        run_id: Unique run ID for the workflow.
        config: Configuration dict with interrupt_on and/or respond settings.
    """
    # Import workflow and get module for event class lookup
    workflow_class, module = import_workflow(workflow_path)

    # Parse config options
    interrupt_on_config = config.get("interrupt_on")
    respond_config = config.get("respond", {})

    # Resolve interrupt config (can be string or dict with condition)
    interrupt_event_class: type[Event] | None = None
    interrupt_condition: dict[str, Any] | None = None
    if interrupt_on_config:
        if isinstance(interrupt_on_config, str):
            interrupt_event_name = interrupt_on_config
        else:
            interrupt_event_name = interrupt_on_config.get("event")
            interrupt_condition = interrupt_on_config.get("condition")
        interrupt_event_class = get_event_class_by_name(module, interrupt_event_name)
        if interrupt_event_class is None:
            print(
                f"ERROR:ValueError:Event class '{interrupt_event_name}' not found in module"
            )
            sys.exit(1)

    # Build response event mapping: {trigger_class: (response_class, fields)}
    response_map: dict[type[Event], tuple[type[Event], dict[str, Any]]] = {}
    for trigger_name, response_info in respond_config.items():
        trigger_class = get_event_class_by_name(module, trigger_name)
        if trigger_class is None:
            print(
                f"ERROR:ValueError:Trigger event class '{trigger_name}' not found in module"
            )
            sys.exit(1)
        response_event_name = response_info.get("event")
        response_fields = response_info.get("fields", {})
        response_class = get_event_class_by_name(module, response_event_name)
        if response_class is None:
            print(
                f"ERROR:ValueError:Response event class '{response_event_name}' not found in module"
            )
            sys.exit(1)
        # Both trigger_class and response_class are narrowed after sys.exit(1) guards
        assert trigger_class is not None
        assert response_class is not None
        response_map[trigger_class] = (response_class, response_fields)

    # Set up DBOS and runtime
    runtime = setup_dbos(db_url)

    # Create workflow instance and launch
    wf = workflow_class(runtime=runtime)
    runtime.launch()

    try:
        ctx = Context(wf)
        handler = ctx._workflow_run(wf, StartEvent(), run_id=run_id)

        async for event in handler.stream_events():
            event_name = type(event).__name__
            print(f"EVENT:{event_name}", flush=True)

            # Check for interrupt condition
            if interrupt_event_class is not None and isinstance(
                event, interrupt_event_class
            ):
                # Check condition fields if present
                should_interrupt = True
                if interrupt_condition:
                    for field, expected_value in interrupt_condition.items():
                        actual_value = getattr(event, field, None)
                        if actual_value != expected_value:
                            should_interrupt = False
                            break
                if should_interrupt:
                    print("INTERRUPTING", flush=True)
                    os._exit(0)

            # Check for response condition (InputRequiredEvent subtypes)
            if isinstance(event, InputRequiredEvent):
                for trigger_class, (response_class, fields) in response_map.items():
                    if isinstance(event, trigger_class):
                        if handler.ctx:
                            response_event = response_class(**fields)
                            handler.ctx.send_event(response_event)
                        break

        result = await handler
        print(f"RESULT:{result}", flush=True)
        print("SUCCESS", flush=True)

    except Exception as e:
        print(f"ERROR:{type(e).__name__}:{e}", flush=True)
        raise

    finally:
        runtime.destroy()


def main() -> None:
    """Entry point for the subprocess runner."""
    parser = argparse.ArgumentParser(
        description="Run workflows in isolated subprocesses for testing"
    )
    parser.add_argument(
        "--workflow",
        required=True,
        help="Module path with class name (e.g., 'tests.fixtures.workflows.hitl:TestWorkflow')",
    )
    parser.add_argument(
        "--db-url",
        required=True,
        help="SQLite database URL",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Unique run ID for the workflow",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="JSON string with configuration",
    )

    args = parser.parse_args()

    config = parse_config(args.config)

    asyncio.run(
        run_workflow(
            workflow_path=args.workflow,
            db_url=args.db_url,
            run_id=args.run_id,
            config=config,
        )
    )


if __name__ == "__main__":
    main()
