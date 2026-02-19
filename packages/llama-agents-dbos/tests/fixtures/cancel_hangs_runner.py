# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Experiment: does cancel_workflow_async actually unblock a waiting control loop?

Tests whether DBOS cancel_workflow_async interrupts a control loop that's
blocked in recv_async (waiting for a tick with a very long timeout), or whether
it hangs until the recv timeout expires.

The key distinction: handle.get_result() just polls the DBOS DB status —
it resolves immediately when cancel marks the status. But the actual asyncio
task running the control loop may still be blocked in recv_async. We test this
by checking whether server shutdown (which awaits the control loop task)
completes in time.

Usage:
    python cancel_hangs_runner.py \
        --workflow "tests.fixtures.sample_workflows.idle_cancel_resume:IdleCancelResumeWorkflow" \
        --db-url "postgresql://user:pass@localhost/db" \
        --run-id "test-cancel-hangs-001" \
        --cancel-wait-timeout 10.0
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dbos import DBOS  # noqa: E402
from llama_agents.dbos.runtime import (  # noqa: E402
    _IO_STREAM_TICK_TOPIC,
    _DBOSInternalWakeUp,
)
from llama_agents.server import WorkflowServer  # noqa: E402
from runner_common import (  # noqa: E402  # ty: ignore[unresolved-import]
    import_workflow,
    setup_dbos,
)


async def run_cancel_hangs_experiment(
    workflow_path: str,
    db_url: str,
    run_id: str,
    cancel_wait_timeout: float,
) -> None:
    """Run the cancel-hangs experiment."""
    workflow_class, module = import_workflow(workflow_path)

    dbos_runtime = setup_dbos(db_url, app_name="test-cancel-hangs")

    wf = workflow_class(runtime=dbos_runtime)
    dbos_runtime.launch()

    store = dbos_runtime.create_workflow_store()
    server_runtime = dbos_runtime.build_server_runtime()

    server = WorkflowServer(
        runtime=server_runtime,
        workflow_store=store,
        idle_timeout=60.0,
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

            # Step 2: Cancel the workflow and send wake-up signal
            # (mirrors what idle_release._release_idle_handler does)
            t0 = time.monotonic()
            await DBOS.cancel_workflow_async(actual_run_id)
            await DBOS.send_async(
                actual_run_id,
                _DBOSInternalWakeUp(),
                topic=_IO_STREAM_TICK_TOPIC,
            )
            print(f"CANCEL_CALLED:elapsed={time.monotonic() - t0:.3f}s", flush=True)

            # Step 3: Check DBOS status (this is just a DB lookup, always fast)
            try:
                status = await DBOS.get_workflow_status_async(actual_run_id)
                print(f"DBOS_STATUS:{status}", flush=True)
            except Exception as e:
                print(f"STATUS_ERROR:{type(e).__name__}:{e}", flush=True)

            # Step 4: Count asyncio tasks to see if the control loop is still alive
            all_tasks = [t for t in asyncio.all_tasks() if not t.done()]
            print(f"LIVE_TASKS_BEFORE_WAIT:{len(all_tasks)}", flush=True)

            # Step 5: Wait and check if the control loop task dies on its own
            await asyncio.sleep(cancel_wait_timeout)

            all_tasks_after = [t for t in asyncio.all_tasks() if not t.done()]
            print(f"LIVE_TASKS_AFTER_WAIT:{len(all_tasks_after)}", flush=True)

            # Print task names/coros for debugging
            for t in all_tasks_after:
                coro = t.get_coro()
                print(f"TASK:{t.get_name()}:{coro}", flush=True)

            print("INNER_DONE", flush=True)

        # If we get here, server.contextmanager() exited cleanly
        print("SERVER_SHUTDOWN_CLEAN", flush=True)

    except Exception as e:
        import traceback

        print(f"ERROR:{type(e).__name__}:{e}", flush=True)
        traceback.print_exc()
    finally:
        print("DESTROYING", flush=True)
        dbos_runtime.destroy()
        print("EXPERIMENT_DONE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment: does cancel_workflow_async unblock recv_async?"
    )
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--cancel-wait-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait after cancel before checking tasks (default 5)",
    )

    args = parser.parse_args()

    asyncio.run(
        run_cancel_hangs_experiment(
            workflow_path=args.workflow,
            db_url=args.db_url,
            run_id=args.run_id,
            cancel_wait_timeout=args.cancel_wait_timeout,
        )
    )


if __name__ == "__main__":
    main()
