# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Subprocess runner for workflow tests with DBOS isolation.

This module provides a CLI runner for executing workflows in isolated
subprocesses, supporting interrupt/resume testing and human-in-the-loop
response simulation.

Usage:
    python /path/to/packages/llama-agents-dbos/tests/fixtures/runner.py \
        --workflow "tests.fixtures.sample_workflows.hitl:TestWorkflow" \
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
import json
import os
import sys
from pathlib import Path
from typing import Any

# Add fixtures dir to find runner_common; safe now that fixtures/workflows
# was renamed to fixtures/sample_workflows to avoid shadowing the real package
sys.path.insert(0, str(Path(__file__).parent))

from runner_common import (  # noqa: E402  # ty: ignore[unresolved-import]
    get_event_class_by_name,
    import_workflow,
    setup_dbos,
)
from workflows.context import Context  # noqa: E402
from workflows.events import Event, InputRequiredEvent, StartEvent  # noqa: E402


async def run_workflow(
    workflow_path: str,
    db_url: str,
    run_id: str,
    config: dict[str, Any],
) -> None:
    """Run the workflow with the specified configuration."""
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
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", default=None)

    args = parser.parse_args()

    asyncio.run(
        run_workflow(
            workflow_path=args.workflow,
            db_url=args.db_url,
            run_id=args.run_id,
            config=json.loads(args.config) if args.config else {},
        )
    )


if __name__ == "__main__":
    main()
