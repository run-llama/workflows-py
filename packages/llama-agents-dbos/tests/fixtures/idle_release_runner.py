# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Subprocess runner for DBOS idle release end-to-end integration test.

Tests the full WorkflowServer + DBOSIdleReleaseDecorator stack:
1. Starts a workflow that goes idle (waits for external event)
2. Waits for WorkflowIdleEvent
3. Waits for idle_timeout to expire so the decorator releases the handler
4. Sends the external event — decorator resumes the workflow automatically
5. Verifies the workflow completes with correct result

Usage:
    python idle_release_runner.py \
        --workflow "tests.fixtures.sample_workflows.idle_cancel_resume:IdleCancelResumeWorkflow" \
        --db-url "postgresql://user:pass@localhost/db" \
        --run-id "test-idle-release-001" \
        --event-type ExternalDataEvent \
        --event-data '{"response": "hello"}' \
        --idle-timeout 0.5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llama_agents.server import WorkflowServer  # noqa: E402
from runner_common import (  # noqa: E402  # ty: ignore[unresolved-import]
    get_event_class_by_name,
    import_workflow,
    setup_dbos,
)


async def run_idle_release(
    workflow_path: str,
    db_url: str,
    run_id: str,
    event_type_name: str,
    event_data_json: str,
    idle_timeout: float,
) -> None:
    """Run the idle release end-to-end test scenario."""
    workflow_class, module = import_workflow(workflow_path)

    dbos_runtime = setup_dbos(db_url, app_name="test-idle-release")

    wf = workflow_class(runtime=dbos_runtime)
    dbos_runtime.launch()

    store = dbos_runtime.create_workflow_store()
    server_runtime = dbos_runtime.build_server_runtime(idle_timeout=idle_timeout)

    server = WorkflowServer(
        runtime=server_runtime,
        workflow_store=store,
        idle_timeout=60.0,  # Server-level timeout (not used since we pass runtime)
    )
    server.add_workflow("test", wf)

    try:
        async with server.contextmanager():
            wf_ref = server._service._runtime.get_workflow("test")
            assert wf_ref is not None

            handler_data = await server._service.start_workflow(wf_ref, run_id)
            actual_run_id = handler_data.run_id
            assert actual_run_id is not None
            print(f"RUN_ID:{actual_run_id}", flush=True)

            # Step 1: Wait for WorkflowIdleEvent
            saw_idle = False
            async for stored_event in store.subscribe_events(actual_run_id):
                event_type = stored_event.event.type
                print(f"EVENT:{event_type}", flush=True)

                if event_type == "WorkflowIdleEvent":
                    saw_idle = True
                    break

            if not saw_idle:
                print("ERROR:Never saw WorkflowIdleEvent", flush=True)
                return

            print("IDLE_DETECTED", flush=True)

            # Step 2: Wait for idle timeout to expire + buffer
            # The decorator's deferred release will cancel the workflow
            wait_time = idle_timeout + 1.0
            print(f"WAITING:{wait_time}s", flush=True)
            await asyncio.sleep(wait_time)
            print("TIMEOUT_ELAPSED", flush=True)

            # Step 3: Send the external event — decorator should auto-resume
            event_class = get_event_class_by_name(module, event_type_name)
            if event_class is None:
                print(f"ERROR:Event class {event_type_name} not found", flush=True)
                return

            event_data = json.loads(event_data_json)
            event_instance = event_class(**event_data)

            await server._service.send_event(
                handler_id=handler_data.handler_id,
                event=event_instance,
            )
            print("EVENT_SENT", flush=True)

            # Step 4: Wait for completion
            async for stored_event in store.subscribe_events(actual_run_id):
                event_type = stored_event.event.type
                print(f"EVENT:{event_type}", flush=True)

                if event_type == "StopEvent":
                    break

            print("SUCCESS", flush=True)

    except Exception as e:
        import traceback

        print(f"ERROR:{type(e).__name__}:{e}", flush=True)
        traceback.print_exc()
        raise
    finally:
        dbos_runtime.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run idle release end-to-end test")
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--event-type", required=True)
    parser.add_argument("--event-data", required=True)
    parser.add_argument("--idle-timeout", type=float, required=True)

    args = parser.parse_args()

    asyncio.run(
        run_idle_release(
            workflow_path=args.workflow,
            db_url=args.db_url,
            run_id=args.run_id,
            event_type_name=args.event_type,
            event_data_json=args.event_data,
            idle_timeout=args.idle_timeout,
        )
    )


if __name__ == "__main__":
    main()
