# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Experiment: does cancel_workflow_async actually unblock a waiting control loop?

Tests whether DBOS cancel_workflow_async interrupts a control loop that's
blocked in recv_async (waiting for a tick with a very long timeout), or whether
it hangs until the recv timeout expires.

The experiment:
1. Starts a workflow that goes idle (blocks in recv_async waiting for event)
2. Waits for WorkflowIdleEvent
3. Calls cancel_workflow_async
4. Measures how long it takes for the DBOS workflow handle to resolve
5. Reports whether it resolved quickly (cancelled properly) or hung

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

            # Step 2: Cancel the workflow and measure how long until the
            # DBOS handle resolves (i.e. the control loop exits)
            t0 = time.monotonic()
            await DBOS.cancel_workflow_async(actual_run_id)
            print(f"CANCEL_CALLED:elapsed={time.monotonic() - t0:.3f}s", flush=True)

            # Step 3: Try to get the workflow result — this will resolve when
            # the control loop actually exits. If recv_async isn't interrupted,
            # this will hang. DBOS may raise DBOSAwaitedWorkflowCancelledError
            # if it detects the cancellation.
            handle = await DBOS.retrieve_workflow_async(actual_run_id)

            try:
                result = await asyncio.wait_for(
                    handle.get_result(),
                    timeout=cancel_wait_timeout,
                )
                elapsed = time.monotonic() - t0
                print(f"CONTROL_LOOP_EXITED:elapsed={elapsed:.3f}s", flush=True)
                print(f"RESULT_TYPE:{type(result).__name__}", flush=True)
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - t0
                print(f"CONTROL_LOOP_HUNG:elapsed={elapsed:.3f}s", flush=True)
                print(
                    f"HUNG:Control loop did not exit within {cancel_wait_timeout}s after cancel",
                    flush=True,
                )
            except Exception as e:
                elapsed = time.monotonic() - t0
                # DBOS raises DBOSAwaitedWorkflowCancelledError when the
                # workflow is cancelled — this means it DID detect the cancel
                # but we need to check if the control loop actually stopped.
                print(
                    f"CONTROL_LOOP_EXITED_WITH_ERROR:elapsed={elapsed:.3f}s:{type(e).__name__}:{e}",
                    flush=True,
                )

            # Step 4: Check DBOS workflow status
            try:
                status = await DBOS.get_workflow_status_async(actual_run_id)
                print(f"DBOS_STATUS:{status}", flush=True)
            except Exception as e:
                print(f"STATUS_ERROR:{type(e).__name__}:{e}", flush=True)

            print("EXPERIMENT_DONE", flush=True)

    except Exception as e:
        import traceback

        print(f"ERROR:{type(e).__name__}:{e}", flush=True)
        traceback.print_exc()
    finally:
        dbos_runtime.destroy()


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
        default=10.0,
        help="Seconds to wait for control loop to exit after cancel (default 10)",
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
