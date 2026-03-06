#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "llama-index-workflows>=2.15.0",
#   "llama-agents-dbos>=0.1.0",
#   "dbos>=2.14.0",
# ]
# ///
"""
Reproduce double-restart hang with the DBOS quickstart counter.

Usage:
    uv run examples/dbos/repro_double_restart.py              # Start new
    uv run examples/dbos/repro_double_restart.py --resume     # Resume after Ctrl+C
    uv run examples/dbos/repro_double_restart.py --clean      # Reset state
    uv run examples/dbos/repro_double_restart.py --journal    # Dump journal table

Steps to reproduce:
    1. uv run examples/dbos/repro_double_restart.py           # Ctrl+C around tick 6
    2. uv run examples/dbos/repro_double_restart.py --resume  # Ctrl+C around tick 12
    3. uv run examples/dbos/repro_double_restart.py --resume  # Should complete or hang
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sqlite3
import threading
import uuid
from pathlib import Path

from dbos import DBOS
from llama_agents.dbos import DBOSRuntime
from pydantic import Field
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent

_DIR = Path(__file__).parent
_DB_FILE = _DIR / ".repro_double_restart.sqlite3"
_RUN_FILE = _DIR / ".repro_run_id"


class Tick(Event):
    count: int = Field(description="Current count")


class CounterResult(StopEvent):
    final_count: int = Field(description="Final counter value")


class CounterWorkflow(Workflow):
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


def dump_journal(run_id: str | None) -> None:
    if not _DB_FILE.exists():
        print("No database file found.")
        return
    conn = sqlite3.connect(str(_DB_FILE))
    try:
        rows = conn.execute(
            "SELECT * FROM workflow_journal ORDER BY seq_num"
        ).fetchall()
    except Exception:
        print("No workflow_journal table found.")
        return
    finally:
        conn.close()

    if not rows:
        print("Journal is empty.")
        return

    pull_count = sum(1 for r in rows if "__pull__" in str(r))
    print(f"Journal: {len(rows)} rows ({pull_count} __pull__)")
    print(f"{'id':>4} | {'run_id':<30} | {'seq':>4} | key")
    print("-" * 70)
    for row in rows[:50]:
        print(f"{row[0]:>4} | {row[1]:<30} | {row[2]:>4} | {row[3]}")
    if len(rows) > 50:
        print(f"  ... ({len(rows) - 50} more rows)")


def _schedule_self_interrupt(after_tick: int) -> None:
    """Monitor stdout for tick count and raise KeyboardInterrupt at the right moment."""
    # We patch the increment step's print to count ticks
    _schedule_self_interrupt._tick_count = 0
    _schedule_self_interrupt._target = after_tick

    original_print = __builtins__["print"] if isinstance(__builtins__, dict) else __builtins__.print

    def counting_print(*args, **kwargs):
        original_print(*args, **kwargs, flush=True)
        msg = " ".join(str(a) for a in args)
        if "[Tick" in msg:
            _schedule_self_interrupt._tick_count += 1
            if _schedule_self_interrupt._tick_count >= _schedule_self_interrupt._target:
                original_print(f"\n>>> Self-interrupting after tick {_schedule_self_interrupt._tick_count}", flush=True)
                # Send SIGINT to ourselves - this is what Ctrl+C does
                os.kill(os.getpid(), signal.SIGINT)

    import builtins
    builtins.print = counting_print


def run(run_id: str, interrupt_at: int | None = None, debug: bool = False) -> None:
    if debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("llama_agents.dbos").setLevel(logging.DEBUG)

    DBOS(
        config={
            "name": "counter-example",
            "system_database_url": f"sqlite+pysqlite:///{_DB_FILE}?check_same_thread=false",
            "run_admin_server": False,
        }
    )

    if interrupt_at is not None:
        _schedule_self_interrupt(interrupt_at)

    runtime = DBOSRuntime()
    workflow = CounterWorkflow(runtime=runtime)

    async def _run() -> None:
        await runtime.launch()
        result = await workflow.run(run_id=run_id)
        print(f"\nResult: final_count = {result.final_count}")

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\nInterrupted — use --resume to continue.")
    finally:
        dump_journal(run_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="DBOS double-restart repro")
    parser.add_argument("--resume", action="store_true", help="Resume last workflow")
    parser.add_argument("--clean", action="store_true", help="Remove state files")
    parser.add_argument("--journal", action="store_true", help="Dump journal table")
    parser.add_argument("--interrupt-at", type=int, default=None,
                        help="Self-interrupt (SIGINT) after N ticks")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.clean:
        for f in [_DB_FILE, _RUN_FILE]:
            if f.exists():
                f.unlink()
                print(f"Removed {f}")
        return

    if args.journal:
        run_id = _RUN_FILE.read_text().strip() if _RUN_FILE.exists() else None
        dump_journal(run_id)
        return

    if args.resume and _RUN_FILE.exists():
        run_id = _RUN_FILE.read_text().strip()
        print(f"Resuming: {run_id}")
    else:
        run_id = f"counter-{uuid.uuid4().hex[:8]}"
        _RUN_FILE.write_text(run_id)
        print(f"Starting: {run_id}")

    run(run_id, interrupt_at=args.interrupt_at, debug=args.debug)


if __name__ == "__main__":
    main()
