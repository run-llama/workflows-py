# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Subprocess runner for DBOS + WorkflowServer integration tests.

Like runner.py but runs through the full WorkflowServer decorator chain
with the store created by DBOSRuntime, enabling end-to-end replay and
event flow testing.

Usage:
    python server_runner.py \
        --workflow "tests.fixtures.sample_workflows.chained:ChainedWorkflow" \
        --db-url "postgresql://user:pass@localhost/db" \
        --run-id "test-001" \
        --check-streams --check-events

    python server_runner.py \
        --workflow "tests.fixtures.sample_workflows.chained:ChainedWorkflow" \
        --db-url "postgresql://user:pass@localhost/db" \
        --run-id "test-001" \
        --config '{"interrupt_on": "StepTwoEvent"}'

Config modes:
    - interrupt_on: Interrupt when event type is seen (uses os._exit(0))
      - String form: "EventName" - interrupt on any instance
      - Dict form: {"event": "EventName", "condition": {"field": value}}
    - run-to-completion: Empty config or omit both fields

Flags:
    --check-streams: After workflow completes, query dbos.streams table
                     and print STREAMS_COUNT:<N>
    --check-events:  After workflow completes, query wf_events table
                     and print EVENTS_COUNT:<N> and EVENT_JSON:<json>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import asyncpg

# Add fixtures dir to find runner_common; safe now that fixtures/workflows
# was renamed to fixtures/sample_workflows to avoid shadowing the real package
sys.path.insert(0, str(Path(__file__).parent))

from llama_agents.server import WorkflowServer  # noqa: E402
from runner_common import (  # noqa: E402  # ty: ignore[unresolved-import]
    get_event_class_by_name,
    import_workflow,
    setup_dbos,
)
from workflows.events import Event  # noqa: E402


async def check_streams_count(db_url: str, run_id: str) -> None:
    """Query DBOS streams table and print count of published_events rows."""
    try:
        dsn = db_url
        if "+psycopg2" in dsn or "+psycopg" in dsn:
            dsn = dsn.split("+")[0] + dsn.split("://", 1)[1]
            dsn = "postgresql://" + dsn.split("://", 1)[1]

        conn = await asyncpg.connect(dsn)
        try:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM dbos.streams "
                "WHERE workflow_uuid = $1 AND key = 'published_events'",
                run_id,
            )
            print(f"STREAMS_COUNT:{count}", flush=True)
        finally:
            await conn.close()
    except Exception as e:
        print(f"ERROR:{type(e).__name__}:Failed to check streams: {e}", flush=True)


async def check_events(db_url: str, run_id: str, schema: str | None = None) -> None:
    """Query wf_events table and print all events."""
    try:
        dsn = db_url
        if "+psycopg2" in dsn or "+psycopg" in dsn:
            dsn = "postgresql://" + dsn.split("://", 1)[1]

        events_table = "wf_events" if schema is None else f"{schema}.wf_events"

        conn = await asyncpg.connect(dsn)
        try:
            rows = await conn.fetch(
                f"SELECT event_json FROM {events_table} "
                f"WHERE run_id = $1 ORDER BY sequence",
                run_id,
            )
            print(f"EVENTS_COUNT:{len(rows)}", flush=True)
            for row in rows:
                print(f"EVENT_JSON:{row['event_json']}", flush=True)
        finally:
            await conn.close()
    except Exception as e:
        print(f"ERROR:{type(e).__name__}:Failed to check events: {e}", flush=True)


async def run_workflow_with_server(
    workflow_path: str,
    db_url: str,
    run_id: str,
    config: dict[str, Any],
    do_check_streams: bool,
    do_check_events: bool,
) -> None:
    """Run workflow through the full WorkflowServer + DBOS decorator chain."""
    workflow_class, module = import_workflow(workflow_path)

    # Parse interrupt config
    interrupt_on_config = config.get("interrupt_on")
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
            print(f"ERROR:ValueError:Event class '{interrupt_event_name}' not found")
            sys.exit(1)

    # Set up DBOS
    dbos_runtime = setup_dbos(db_url, app_name="test-server-workflow")

    # Create workflow instance, register with DBOS runtime
    wf = workflow_class(runtime=dbos_runtime)
    dbos_runtime.launch()

    # Create store and build server runtime
    store = dbos_runtime.create_workflow_store()
    await store.start()  # type: ignore[attr-defined]
    await store.run_migrations()  # type: ignore[attr-defined]

    server_runtime = dbos_runtime.build_server_runtime()

    server = WorkflowServer(
        runtime=server_runtime,
        workflow_store=store,
        idle_timeout=60.0,
    )
    server.add_workflow("test", wf)

    schema = getattr(store, "_schema", None)

    try:
        async with server.contextmanager():
            wf_ref = server._service._runtime.get_workflow("test")
            assert wf_ref is not None

            handler_data = await server._service.start_workflow(wf_ref, run_id)
            actual_run_id = handler_data.run_id
            assert actual_run_id is not None

            # Stream events from the store
            async for stored_event in store.subscribe_events(actual_run_id):
                envelope = stored_event.event
                event_type = envelope.type
                print(f"EVENT:{event_type}", flush=True)

                # Check for interrupt: match on event type name
                if interrupt_event_class is not None:
                    types = (envelope.types or []) + [envelope.type]
                    if interrupt_event_class.__name__ in types:
                        should_interrupt = True
                        if interrupt_condition:
                            data = envelope.value or {}
                            for field, expected_value in interrupt_condition.items():
                                if data.get(field) != expected_value:
                                    should_interrupt = False
                                    break
                        if should_interrupt:
                            print("INTERRUPTING", flush=True)
                            os._exit(0)

            print("SUCCESS", flush=True)

            if do_check_streams:
                await check_streams_count(db_url, actual_run_id)

            if do_check_events:
                await check_events(db_url, actual_run_id, schema)

    except Exception as e:
        print(f"ERROR:{type(e).__name__}:{e}", flush=True)
        raise
    finally:
        await store.close()  # type: ignore[attr-defined]
        dbos_runtime.destroy()


def main() -> None:
    """Entry point for the subprocess runner."""
    parser = argparse.ArgumentParser(
        description="Run workflows through WorkflowServer + DBOS for testing"
    )
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--check-streams", action="store_true")
    parser.add_argument("--check-events", action="store_true")

    args = parser.parse_args()

    asyncio.run(
        run_workflow_with_server(
            workflow_path=args.workflow,
            db_url=args.db_url,
            run_id=args.run_id,
            config=json.loads(args.config) if args.config else {},
            do_check_streams=args.check_streams,
            do_check_events=args.check_events,
        )
    )


if __name__ == "__main__":
    main()
