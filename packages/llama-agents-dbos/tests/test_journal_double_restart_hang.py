# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Reproduce the journal hang on second Ctrl+C restart.

Bug: after two interrupt-restart cycles, the workflow hangs forever.

The workflow_journal table accumulates entries across restarts. Events that
flow through DBOS.send/recv create __pull__ entries in the journal. After
two restarts, the accumulated journal entries cause DBOS's function_id
tracking to diverge, and recv_async blocks forever instead of replaying.

This test uses the subprocess runner pattern to realistically simulate
two interrupt-restart cycles with proper DBOS process isolation.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    assert_no_determinism_errors,  # pyright: ignore[reportAttributeAccessIssue]
    run_scenario,  # pyright: ignore[reportAttributeAccessIssue]
    run_simple_counter,  # pyright: ignore[reportAttributeAccessIssue]
)


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    return tmp_path / "double_restart_test.sqlite3"


def _get_journal_rows(db_path: Path, run_id: str) -> list[tuple[Any, ...]]:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT * FROM workflow_journal WHERE run_id=? ORDER BY seq_num",
            (run_id,),
        ).fetchall()
    finally:
        conn.close()


def _log_journal(db_path: Path, run_id: str, label: str) -> None:
    """Print journal stats for debugging — no assertions."""
    rows = _get_journal_rows(db_path, run_id)
    pull_count = sum(1 for r in rows if "__pull__" in r[3])
    print(f"{label}: {len(rows)} journal rows ({pull_count} __pull__)")


COUNTER_WORKFLOW = "tests.fixtures.sample_workflows.counter:CounterWorkflow"
RESPOND_CONFIG = {
    "respond": {
        "CounterTickEvent": {
            "event": "CounterContinueEvent",
            "fields": {},
        }
    }
}


def _run_double_restart_simple_counter(
    db_path: Path,
    run_id: str,
    fast_polling: bool = True,
    call_close: bool = False,
) -> None:
    """Three-phase double-restart test for the simple (non-HITL) counter.

    Run 1: start, interrupt at tick 5
    Run 2: resume, interrupt at tick 12
    Run 3: resume, should complete to 20
    """
    db_url = f"sqlite+pysqlite:///{db_path}?check_same_thread=false"

    # --- Run 1: start, interrupt at tick 5 ---
    result1 = run_simple_counter(
        db_url=db_url,
        run_id=run_id,
        interrupt_at=5,
        fast_polling=fast_polling,
        call_close=call_close,
    )
    assert "STEP:start:complete" in result1.stdout, (
        f"Start step should complete.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
    )
    assert "INTERRUPTING" in result1.stdout, (
        f"Should interrupt at tick 5.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
    )
    if call_close:
        assert "CLOSE_CALLED" in result1.stdout
    _log_journal(db_path, run_id, "After run 1")

    # --- Run 2: resume, interrupt at tick 12 ---
    result2 = run_simple_counter(
        db_url=db_url,
        run_id=run_id,
        interrupt_at=12,
        fast_polling=fast_polling,
        call_close=call_close,
    )
    assert "INTERRUPTING" in result2.stdout, (
        f"Should interrupt at tick 12.\nstdout: {result2.stdout}\nstderr: {result2.stderr}"
    )
    if call_close:
        assert "CLOSE_CALLED" in result2.stdout
    _log_journal(db_path, run_id, "After run 2")

    # --- Run 3: resume, should complete to 20 ---
    result3 = run_simple_counter(
        db_url=db_url,
        run_id=run_id,
        fast_polling=fast_polling,
        timeout=30.0,
    )
    _log_journal(db_path, run_id, "After run 3")

    assert_no_determinism_errors(result3)
    assert "SUCCESS" in result3.stdout, (
        f"Should complete successfully.\n"
        f"stdout: {result3.stdout}\nstderr: {result3.stderr}"
    )


def test_double_restart_counter_workflow(test_db_path: Path) -> None:
    """Two interrupt-restart cycles on a HITL counter workflow.

    Run 1: Start counter, interrupt at tick 5
    Run 2: Resume, interrupt at tick 15
    Run 3: Resume, should complete to tick 40 — but may hang due to
            journal accumulation causing DBOS function_id desync.

    The counter uses HITL events flowing through DBOS.send/recv, which
    produces __pull__ entries in the journal.
    """
    run_id = "counter-double-restart"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    # --- Run 1: start, interrupt at tick 5 ---
    result1 = run_scenario(
        workflow=COUNTER_WORKFLOW,
        db_url=db_url,
        run_id=run_id,
        config={
            **RESPOND_CONFIG,
            "interrupt_on": {"event": "CounterTickEvent", "condition": {"count": 5}},
        },
    )
    assert "STEP:start:complete" in result1.stdout, (
        f"Start step should complete.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
    )
    assert "INTERRUPTING" in result1.stdout, (
        f"Should interrupt at tick 5.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
    )
    _log_journal(test_db_path, run_id, "After run 1")

    # --- Run 2: resume, interrupt at tick 15 ---
    result2 = run_scenario(
        workflow=COUNTER_WORKFLOW,
        db_url=db_url,
        run_id=run_id,
        config={
            **RESPOND_CONFIG,
            "interrupt_on": {"event": "CounterTickEvent", "condition": {"count": 15}},
        },
    )
    assert "INTERRUPTING" in result2.stdout, (
        f"Should interrupt at tick 15.\nstdout: {result2.stdout}\nstderr: {result2.stderr}"
    )
    _log_journal(test_db_path, run_id, "After run 2")

    # --- Run 3: resume, should complete to 40 ---
    result3 = run_scenario(
        workflow=COUNTER_WORKFLOW,
        db_url=db_url,
        run_id=run_id,
        config=RESPOND_CONFIG,
        timeout=15.0,
    )
    _log_journal(test_db_path, run_id, "After run 3")

    assert_no_determinism_errors(result3)
    assert "SUCCESS" in result3.stdout, (
        f"Should complete successfully after second resume.\n"
        f"stdout: {result3.stdout}\nstderr: {result3.stderr}"
    )


def test_double_restart_simple_counter_fast_polling(test_db_path: Path) -> None:
    """Two interrupt-restart cycles on the non-HITL quickstart counter.

    Uses fast polling (test default) — no HITL, plain Tick events.
    """
    _run_double_restart_simple_counter(
        test_db_path, run_id="simple-counter-fast", fast_polling=True
    )


def test_double_restart_with_close(test_db_path: Path) -> None:
    """Two interrupt cycles where adapter.close() is called before exit.

    This is the actual bug scenario: close() on the old code called
    DBOS.send(_DBOSInternalShutdown, ...) which persisted a poison message
    to the notifications table. On resume, DBOS replays that notification,
    which poisons the adapter's _closed flag. After two such cycles, the
    accumulated __pull__ journal entries cause DBOS's function_id tracking
    to diverge, and recv_async blocks forever.
    """
    _run_double_restart_simple_counter(
        test_db_path, run_id="close-double-restart", call_close=True
    )


def test_double_restart_simple_counter_default_polling(test_db_path: Path) -> None:
    """Same as fast_polling variant but with DEFAULT DBOS polling intervals.

    The user's quickstart example uses DBOSRuntime() with no polling config,
    meaning DBOS uses its default notification_listener_polling_interval_sec.
    """
    _run_double_restart_simple_counter(
        test_db_path, run_id="simple-counter-default", fast_polling=False
    )
