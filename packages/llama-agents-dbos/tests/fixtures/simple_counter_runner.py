# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Runner for simple (non-HITL) counter workflow.

Mimics the user's durable_workflow.py pattern: runs the counter and
interrupts after a specified tick count.

Three interrupt modes:
  - Hard (default): os._exit(0) — skips all teardown, like kill -9
  - Graceful (--graceful-interrupt): sends SIGINT to self, which raises
    KeyboardInterrupt. The finally block calls runtime.destroy() →
    adapter.close(). This is what matters for the double-restart bug:
    close() used to persist a poisoned shutdown message via DBOS.send().
  - Call-close (--call-close): creates a fresh adapter and calls close()
    while DBOS is still fully alive, then os._exit(0). This reliably
    triggers the old bug where close() persisted a _DBOSInternalShutdown
    message via DBOS.send().

Usage:
    python simple_counter_runner.py \
        --db-url "sqlite+pysqlite:///path/to/db" \
        --run-id "test-001" \
        [--interrupt-at 5] \
        [--target 20] \
        [--fast-polling] \
        [--graceful-interrupt] \
        [--call-close]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any

# Add fixtures dir to find runner_common
sys.path.insert(0, str(Path(__file__).parent))

import runner_common  # noqa: E402, F401  # ty: ignore[unresolved-import]  # side-effect: sys.path setup
from dbos import DBOS, DBOSConfig  # noqa: E402
from llama_agents.dbos import DBOSRuntime  # noqa: E402
from sample_workflows.simple_counter import (  # noqa: E402  # ty: ignore[unresolved-import]
    SimpleCounterWorkflow,
)
from workflows.events import StartEvent  # noqa: E402


async def run(
    db_url: str,
    run_id: str,
    target: int,
    fast_polling: bool,
) -> None:
    config: DBOSConfig = {
        "name": "simple-counter-test",
        "system_database_url": db_url,
        "run_admin_server": False,
    }
    if fast_polling:
        config["notification_listener_polling_interval_sec"] = 0.01

    DBOS(config=config)

    runtime_kwargs: dict = {}
    if fast_polling:
        runtime_kwargs["polling_interval_sec"] = 0.01

    runtime = DBOSRuntime(**runtime_kwargs)
    wf = SimpleCounterWorkflow(target=target, runtime=runtime)
    await runtime.launch()

    try:
        # Check for existing workflow (resume case)
        existing = await DBOS.get_workflow_status_async(run_id)
        if existing is not None:
            print(f"RESUMING:{run_id}", flush=True)
            handler = wf.run(run_id=run_id)
        else:
            print(f"STARTING:{run_id}", flush=True)
            handler = wf.run(start_event=StartEvent(), run_id=run_id)

        result = await handler
        print(f"RESULT:{result}", flush=True)
        print("SUCCESS", flush=True)

    except Exception as e:
        print(f"ERROR:{type(e).__name__}:{e}", flush=True)
        raise

    finally:
        await runtime.destroy()


# Monkey-patch the workflow's increment step to intercept ticks
_original_print = print


def _call_adapter_close(run_id: str) -> None:
    """Create a fresh adapter and call close() while DBOS is still alive.

    On old code, close() calls DBOS.send(_DBOSInternalShutdown, ...) which
    persists a poison message to the notifications table.
    On new code, close() just sets a process-local asyncio.Event.
    """
    from llama_agents.dbos.runtime import InternalDBOSAdapter

    adapter = InternalDBOSAdapter(
        run_id=run_id,
        engine=None,  # type: ignore[arg-type]
        state_type=None,
    )

    def _run() -> None:
        asyncio.run(adapter.close())

    t = threading.Thread(target=_run)
    t.start()
    t.join(timeout=5)


def _intercepting_print(*args: object, **kwargs: Any) -> None:
    """Intercept print calls to detect tick count for interruption."""
    _original_print(*args, **kwargs)
    if args and isinstance(args[0], str) and args[0].startswith("STEP:increment:"):
        count = int(str(args[0]).split(":")[-1])
        interrupt_at = getattr(_intercepting_print, "_interrupt_at", None)
        if interrupt_at is not None and count >= interrupt_at:
            _original_print("INTERRUPTING", flush=True)
            if getattr(_intercepting_print, "_call_close", False):
                # Call adapter.close() while DBOS is still alive, then hard-exit.
                # On old code this poisons the DB; on new code it's harmless.
                run_id = getattr(_intercepting_print, "_run_id", None)
                if run_id:
                    _call_adapter_close(run_id)
                    _original_print("CLOSE_CALLED", flush=True)
                os._exit(0)
            elif getattr(_intercepting_print, "_graceful", False):
                # Send SIGINT to ourselves — KeyboardInterrupt unwinds through
                # asyncio.run(), the finally block calls runtime.destroy() which
                # calls close() on adapters. This is what Ctrl+C does.
                os.kill(os.getpid(), signal.SIGINT)
            else:
                os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--interrupt-at", type=int, default=None)
    parser.add_argument("--target", type=int, default=20)
    parser.add_argument("--fast-polling", action="store_true")
    parser.add_argument("--graceful-interrupt", action="store_true")
    parser.add_argument("--call-close", action="store_true")
    args = parser.parse_args()

    if args.interrupt_at is not None:
        _intercepting_print._interrupt_at = args.interrupt_at  # type: ignore[attr-defined]
        _intercepting_print._graceful = args.graceful_interrupt  # type: ignore[attr-defined]
        _intercepting_print._call_close = args.call_close  # type: ignore[attr-defined]
        _intercepting_print._run_id = args.run_id  # type: ignore[attr-defined]
        import builtins

        builtins.print = _intercepting_print  # type: ignore[assignment]

    try:
        asyncio.run(
            run(
                db_url=args.db_url,
                run_id=args.run_id,
                target=args.target,
                fast_polling=args.fast_polling,
            )
        )
    except KeyboardInterrupt:
        _original_print("GRACEFUL_SHUTDOWN", flush=True)


if __name__ == "__main__":
    main()
