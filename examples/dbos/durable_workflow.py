#!/usr/bin/env python3
"""
DBOS Counter Example

Simple looping workflow that increments a counter until it reaches 20.

Usage:
    python -m examples.dbos_durability.counter_example              # Start new
    python -m examples.dbos_durability.counter_example --resume     # Resume last
    python -m examples.dbos_durability.counter_example --clean      # Reset state

Try Ctrl+C mid-run to test resume behavior.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import threading
import time
import uuid
from pathlib import Path

from dbos import DBOS
from llama_agents.dbos import DBOSRuntime
from pydantic import Field
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent

_DIR = Path(__file__).parent
_DB_FILE = _DIR / ".dbos_data.sqlite3"
_RUN_FILE = _DIR / ".last_run_id"


class Tick(Event):
    count: int = Field(description="Current count")


class CounterResult(StopEvent):
    final_count: int = Field(description="Final counter value")


class CounterWorkflow(Workflow):
    """Looping counter workflow - increments until reaching 20."""

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> Tick:
        await ctx.store.set("count", 0)
        print("[Start] Initializing counter to 0")
        return Tick(count=0)

    @step
    async def increment(self, ctx: Context, ev: Tick) -> Tick | CounterResult:
        count = ev.count + 1
        await ctx.store.set("count", count)
        print(f"[Tick {count:2d}] count = {count}")

        if count >= 20:
            return CounterResult(final_count=count)

        await asyncio.sleep(0.5)
        return Tick(count=count)


def run(run_id: str) -> None:
    """Run the counter workflow."""
    DBOS(
        config={
            "name": "counter-example",
            "system_database_url": f"sqlite+pysqlite:///{_DB_FILE}?check_same_thread=false",
            "run_admin_server": False,
        }
    )

    runtime = DBOSRuntime()
    workflow = CounterWorkflow(runtime=runtime)
    runtime.launch()

    interrupted = False

    def handle_sigint(signum: int, frame: object) -> None:
        nonlocal interrupted
        if interrupted:
            # Second Ctrl+C - force exit
            os._exit(130)
        interrupted = True
        print("\nInterrupted - workflow state saved. Use --resume to continue.")

        def delayed_exit() -> None:
            time.sleep(0.1)
            os._exit(130)

        threading.Thread(target=delayed_exit, daemon=True).start()

    # Install signal handler before running
    signal.signal(signal.SIGINT, handle_sigint)

    async def _run() -> None:
        result = await workflow.run(run_id=run_id)
        print(f"\nResult: final_count = {result.final_count}")

    try:
        asyncio.run(_run())
    except (KeyboardInterrupt, SystemExit):
        pass  # Already handled by signal handler
    finally:
        if not interrupted:
            try:
                runtime.destroy()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="DBOS Counter Example")
    parser.add_argument("--resume", action="store_true", help="Resume last workflow")
    parser.add_argument("--clean", action="store_true", help="Remove state files")
    args = parser.parse_args()

    if args.clean:
        for f in [_DB_FILE, _RUN_FILE]:
            if f.exists():
                f.unlink()
                print(f"Removed {f}")
        return

    if args.resume and _RUN_FILE.exists():
        run_id = _RUN_FILE.read_text().strip()
        print(f"Resuming: {run_id}")
    else:
        run_id = f"counter-{uuid.uuid4().hex[:8]}"
        _RUN_FILE.write_text(run_id)
        print(f"Starting: {run_id}")

    run(run_id)


if __name__ == "__main__":
    main()
