# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Runner for simple (non-HITL) counter workflow.

Mimics the user's durable_workflow.py pattern: runs the counter and uses
os._exit() to simulate Ctrl+C after a specified tick count.

Usage:
    python simple_counter_runner.py \
        --db-url "sqlite+pysqlite:///path/to/db" \
        --run-id "test-001" \
        [--interrupt-at 5] \
        [--target 20] \
        [--fast-polling]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add necessary source paths
_DBOS_PACKAGE_DIR = Path(__file__).parent.parent.parent
_SYS_PATHS = [
    str(_DBOS_PACKAGE_DIR),
    str(_DBOS_PACKAGE_DIR / "src"),
    str(_DBOS_PACKAGE_DIR.parent / "llama-index-workflows" / "src"),
    str(_DBOS_PACKAGE_DIR.parent / "llama-agents-server" / "src"),
    str(_DBOS_PACKAGE_DIR.parent / "llama-agents-client" / "src"),
    str(_DBOS_PACKAGE_DIR.parent / "llama-index-instrumentation" / "src"),
]
for _p in _SYS_PATHS:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dbos import DBOS  # noqa: E402
from llama_agents.dbos import DBOSRuntime  # noqa: E402
from sample_workflows.simple_counter import SimpleCounterWorkflow  # noqa: E402
from workflows.events import StartEvent  # noqa: E402


async def run(
    db_url: str,
    run_id: str,
    interrupt_at: int | None,
    target: int,
    fast_polling: bool,
) -> None:
    config: dict = {
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


def _intercepting_print(*args: object, **kwargs: object) -> None:
    """Intercept print calls to detect tick count for interruption."""
    _original_print(*args, **kwargs)
    if args and isinstance(args[0], str) and args[0].startswith("STEP:increment:"):
        count = int(str(args[0]).split(":")[-1])
        interrupt_at = getattr(_intercepting_print, "_interrupt_at", None)
        if interrupt_at is not None and count >= interrupt_at:
            _original_print("INTERRUPTING", flush=True)
            os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--interrupt-at", type=int, default=None)
    parser.add_argument("--target", type=int, default=20)
    parser.add_argument("--fast-polling", action="store_true")
    args = parser.parse_args()

    if args.interrupt_at is not None:
        _intercepting_print._interrupt_at = args.interrupt_at  # type: ignore[attr-defined]
        import builtins

        builtins.print = _intercepting_print  # type: ignore[assignment]

    asyncio.run(
        run(
            db_url=args.db_url,
            run_id=args.run_id,
            interrupt_at=args.interrupt_at,
            target=args.target,
            fast_polling=args.fast_polling,
        )
    )


if __name__ == "__main__":
    main()
