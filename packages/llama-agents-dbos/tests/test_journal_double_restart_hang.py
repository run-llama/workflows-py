# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Regression: HITL workflow must survive two interrupt-restart cycles."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    assert_no_determinism_errors,  # pyright: ignore[reportAttributeAccessIssue]
    run_scenario,  # pyright: ignore[reportAttributeAccessIssue]
)

COUNTER_WORKFLOW = "tests.fixtures.sample_workflows.counter:CounterWorkflow"
RESPOND_CONFIG = {
    "respond": {
        "CounterTickEvent": {
            "event": "CounterContinueEvent",
            "fields": {},
        }
    }
}


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
    rows = _get_journal_rows(db_path, run_id)
    pull_count = sum(1 for r in rows if "__pull__" in r[3])
    print(f"{label}: {len(rows)} journal rows ({pull_count} __pull__)")


def _run_double_restart(
    db_path: Path,
    run_id: str,
    call_close: bool = False,
) -> None:
    db_url = f"sqlite+pysqlite:///{db_path}?check_same_thread=false"

    # --- Run 1: start, interrupt at tick 5 ---
    result1 = run_scenario(
        workflow=COUNTER_WORKFLOW,
        db_url=db_url,
        run_id=run_id,
        config={
            **RESPOND_CONFIG,
            "interrupt_on": {"event": "CounterTickEvent", "condition": {"count": 5}},
        },
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

    # --- Run 2: resume, interrupt at tick 15 ---
    result2 = run_scenario(
        workflow=COUNTER_WORKFLOW,
        db_url=db_url,
        run_id=run_id,
        config={
            **RESPOND_CONFIG,
            "interrupt_on": {"event": "CounterTickEvent", "condition": {"count": 15}},
        },
        call_close=call_close,
    )
    assert "INTERRUPTING" in result2.stdout, (
        f"Should interrupt at tick 15.\nstdout: {result2.stdout}\nstderr: {result2.stderr}"
    )
    if call_close:
        assert "CLOSE_CALLED" in result2.stdout
    _log_journal(db_path, run_id, "After run 2")

    # --- Run 3: resume, should complete to 40 ---
    result3 = run_scenario(
        workflow=COUNTER_WORKFLOW,
        db_url=db_url,
        run_id=run_id,
        config=RESPOND_CONFIG,
        timeout=15.0,
    )
    _log_journal(db_path, run_id, "After run 3")

    assert_no_determinism_errors(result3)
    assert "SUCCESS" in result3.stdout, (
        f"Should complete successfully.\n"
        f"stdout: {result3.stdout}\nstderr: {result3.stderr}"
    )


def test_double_restart_counter_workflow(test_db_path: Path) -> None:
    _run_double_restart(test_db_path, run_id="counter-double-restart")


def test_double_restart_with_close(test_db_path: Path) -> None:
    """Same as above, but calls adapter.close() before each exit."""
    _run_double_restart(test_db_path, run_id="close-double-restart", call_close=True)
