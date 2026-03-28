# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for orphan purge: orphaned DBOS operations are cleaned up on recovery."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    run_scenario,  # pyright: ignore[reportAttributeAccessIssue]
)

SLOW_FANOUT_WORKFLOW = (
    "tests.fixtures.sample_workflows.slow_fan_out_hitl:SlowFanOutWorkflow"
)
RESPOND_CONFIG: dict[str, Any] = {
    "respond": {
        "SlowFanOutTickEvent": {
            "event": "SlowFanOutContinueEvent",
            "fields": {},
        }
    }
}
INTERRUPT_CONFIG: dict[str, Any] = {
    "respond": RESPOND_CONFIG["respond"],
    "interrupt_on": {"event": "SlowFanOutTickEvent", "condition": {"round": 1}},
}


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    return tmp_path / "orphan_determinism_test.sqlite3"


def _db_url(db_path: Path) -> str:
    return f"sqlite+pysqlite:///{db_path}?check_same_thread=false"


def _query_scalar(db_path: Path, sql: str, params: tuple[Any, ...] = ()) -> Any:
    """Run a single-value query against the test DB, returning None if table missing."""
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else None
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


def _get_max_fid(db_path: Path, run_id: str) -> int:
    result = _query_scalar(
        db_path,
        "SELECT MAX(function_id) FROM operation_outputs WHERE workflow_uuid=?",
        (run_id,),
    )
    return result or 0


def _inject_orphaned_operation(
    db_path: Path,
    run_id: str,
    function_id: int,
    function_name: str,
    output: str = "null",
) -> None:
    """Insert a fake orphaned operation to simulate crash-left rows."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT INTO operation_outputs "
            "(workflow_uuid, function_id, function_name, output, started_at_epoch_ms) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, function_id, function_name, output, 0),
        )
        conn.commit()
    finally:
        conn.close()


def _get_function_at_fid(db_path: Path, run_id: str, fid: int) -> str | None:
    return _query_scalar(
        db_path,
        "SELECT function_name FROM operation_outputs "
        "WHERE workflow_uuid=? AND function_id=?",
        (run_id, fid),
    )


def _run_interrupt_and_get_max_fid(db_path: Path, run_id: str) -> int:
    """Run the workflow, interrupt on round 1, return max_fid."""
    result = run_scenario(
        workflow=SLOW_FANOUT_WORKFLOW,
        db_url=_db_url(db_path),
        run_id=run_id,
        config=INTERRUPT_CONFIG,
        timeout=30.0,
    )
    assert "INTERRUPTING" in result.stdout, (
        f"Run should have interrupted.\nstdout: {result.stdout}"
    )
    return _get_max_fid(db_path, run_id)


def test_injected_orphan_is_purged_on_recovery(test_db_path: Path) -> None:
    """Injected orphaned operations are purged on recovery — recovery succeeds
    despite mismatched function_names at fids beyond the journal boundary."""
    run_id = "injected-orphan-test"
    orphan_name = "FAKE_ORPHANED_STEP"

    max_fid = _run_interrupt_and_get_max_fid(test_db_path, run_id)

    orphan_fid = max_fid + 10
    _inject_orphaned_operation(
        test_db_path, run_id, orphan_fid, function_name=orphan_name
    )

    result = run_scenario(
        workflow=SLOW_FANOUT_WORKFLOW,
        db_url=_db_url(test_db_path),
        run_id=run_id,
        config=RESPOND_CONFIG,
        timeout=30.0,
    )

    combined = result.stdout + result.stderr
    assert "SUCCESS" in result.stdout, (
        f"Recovery should succeed after purging orphan.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "DBOSUnexpectedStepError" not in combined, (
        "Should NOT get determinism error after purge"
    )

    orphan_fn = _get_function_at_fid(test_db_path, run_id, orphan_fid)
    assert orphan_fn != orphan_name, (
        f"Orphaned operation at fid {orphan_fid} should have been purged, "
        f"but found function_name={orphan_fn}"
    )
