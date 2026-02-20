# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Test runner: does idle-release cancel actually unblock the control loop?

Verifies that after the idle release decorator cancels a workflow, the control
loop task (blocked in recv_async) actually exits — not just that the DBOS status
changes. This exercises the real production code path through
DBOSIdleReleaseDecorator._release_idle_handler().

Usage:
    python cancel_hangs_runner.py \
        --workflow "tests.fixtures.sample_workflows.idle_cancel_resume:IdleCancelResumeWorkflow" \
        --db-url "postgresql://user:pass@localhost/db" \
        --run-id "test-cancel-hangs-001" \
        --idle-timeout 0.5 \
        --cancel-wait-timeout 5.0
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llama_agents.server import WorkflowServer  # noqa: E402
from runner_common import (  # noqa: E402  # ty: ignore[unresolved-import]
    import_workflow,
    setup_dbos,
)


async def run_cancel_hangs_experiment(
    workflow_path: str,
    db_url: str,
    run_id: str,
    idle_timeout: float,
    cancel_wait_timeout: float,
) -> None:
    """Run the cancel-hangs experiment using the real idle release path."""
    workflow_class, module = import_workflow(workflow_path)

    dbos_runtime = setup_dbos(db_url, app_name="test-cancel-hangs")

    wf = workflow_class(runtime=dbos_runtime)
    dbos_runtime.launch()

    store = dbos_runtime.create_workflow_store()
    # Use build_server_runtime with idle_timeout so the idle release decorator
    # is wired up — this is the real production code path.
    server_runtime = dbos_runtime.build_server_runtime(idle_timeout=idle_timeout)

    server = WorkflowServer(
        runtime=server_runtime,
        workflow_store=store,
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

            # Step 2: Wait for idle timeout to expire + cancel_wait_timeout for
            # the control loop to actually exit after the wake-up signal
            total_wait = idle_timeout + cancel_wait_timeout
            print(f"WAITING:{total_wait}s", flush=True)
            await asyncio.sleep(total_wait)
            print("WAIT_ELAPSED", flush=True)

            # Step 3: Check if control loop task is gone
            live_tasks = [t for t in asyncio.all_tasks() if not t.done()]
            zombie_tasks = [
                t
                for t in live_tasks
                if "_single_pull" in repr(t.get_coro())
                or "_execute_workflow" in repr(t.get_coro())
            ]
            print(f"LIVE_TASKS:{len(live_tasks)}", flush=True)
            print(f"ZOMBIE_TASKS:{len(zombie_tasks)}", flush=True)
            for t in live_tasks:
                print(f"TASK:{t.get_name()}:{t.get_coro()}", flush=True)

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
        description="Test: does idle-release cancel unblock the control loop?"
    )
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=0.5,
        help="Idle timeout in seconds (default 0.5)",
    )
    parser.add_argument(
        "--cancel-wait-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait after release before checking tasks (default 5)",
    )

    args = parser.parse_args()

    asyncio.run(
        run_cancel_hangs_experiment(
            workflow_path=args.workflow,
            db_url=args.db_url,
            run_id=args.run_id,
            idle_timeout=args.idle_timeout,
            cancel_wait_timeout=args.cancel_wait_timeout,
        )
    )


if __name__ == "__main__":
    main()
