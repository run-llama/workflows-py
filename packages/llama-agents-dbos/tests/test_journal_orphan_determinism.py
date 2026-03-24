# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Reproduction: orphaned DBOS operations cause DBOSUnexpectedStepError.

After a crash, committed operation_outputs rows can persist beyond the journal
boundary. If the fresh execution's fid sequence diverges (due to non-deterministic
done.pop() in wait_for_next_task), DBOS detects the function_name mismatch and
raises DBOSUnexpectedStepError.
"""

from __future__ import annotations

import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    RUNNER_PATH,  # pyright: ignore[reportAttributeAccessIssue]
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


def _get_journal_count(db_path: Path, run_id: str) -> int:
    result = _query_scalar(
        db_path,
        "SELECT COUNT(*) FROM workflow_journal WHERE run_id=?",
        (run_id,),
    )
    return result or 0


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


def _run_and_kill_after(
    db_path: Path,
    run_id: str,
    kill_after_sec: float,
) -> subprocess.CompletedProcess[str]:
    """Run a workflow subprocess and SIGKILL it after a delay."""
    import json
    import sys

    cmd = [
        sys.executable,
        RUNNER_PATH,
        "--workflow",
        SLOW_FANOUT_WORKFLOW,
        "--db-url",
        _db_url(db_path),
        "--run-id",
        run_id,
        "--config",
        json.dumps(RESPOND_CONFIG),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(kill_after_sec)
    proc.kill()
    stdout, stderr = proc.communicate(timeout=5)
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout=stdout.decode() if isinstance(stdout, bytes) else (stdout or ""),
        stderr=stderr.decode() if isinstance(stderr, bytes) else (stderr or ""),
    )


def _has_determinism_error(result: subprocess.CompletedProcess[str]) -> bool:
    combined = result.stdout + result.stderr
    return "DBOSUnexpectedStepError" in combined or "Error 11" in combined


@pytest.mark.parametrize(
    "orphan_name,extra_assert",
    [
        ("FAKE_ORPHANED_STEP", None),
        (
            "tests.fixtures.sample_workflows.slow_fan_out_hitl."
            "SlowFanOutWorkflow.worker_alpha",
            # The error message should show the mismatch between orphan and actual call
            lambda combined: (
                "worker_alpha" in combined and "_durable_time" in combined
            ),
        ),
    ],
    ids=["generic-orphan", "realistic-worker-orphan"],
)
def test_injected_orphan_triggers_determinism_error(
    test_db_path: Path,
    orphan_name: str,
    extra_assert: Any,
) -> None:
    """Orphaned operations with mismatched function_names cause
    DBOSUnexpectedStepError on recovery."""
    run_id = "injected-orphan-test"

    max_fid = _run_interrupt_and_get_max_fid(test_db_path, run_id)

    # Inject orphan at the next fid (will be _durable_time in fresh execution)
    orphan_fid = max_fid + 1
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
    assert _has_determinism_error(result), (
        f"Expected DBOSUnexpectedStepError but got exit code {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert orphan_name in combined, (
        f"Error should reference the injected function name: {orphan_name}"
    )
    if extra_assert is not None:
        assert extra_assert(combined), (
            "Error should show function_name mismatch between orphan and actual call"
        )


def test_sigkill_creates_orphaned_operations(test_db_path: Path) -> None:
    """SIGKILL mid-execution creates orphaned DBOS operations beyond the
    journal boundary — the precondition for the determinism bug."""
    run_id = "sigkill-orphan-check"

    max_fid_1 = _run_interrupt_and_get_max_fid(test_db_path, run_id)
    journal_1 = _get_journal_count(test_db_path, run_id)

    _run_and_kill_after(test_db_path, run_id, kill_after_sec=1.5)

    max_fid_2 = _get_max_fid(test_db_path, run_id)
    journal_2 = _get_journal_count(test_db_path, run_id)

    assert max_fid_2 > max_fid_1, "SIGKILL should have added more DBOS operations"
    assert journal_2 > journal_1, "SIGKILL should have added more journal entries"

    # Normal recovery should succeed (same process = same done.pop() order)
    result = run_scenario(
        workflow=SLOW_FANOUT_WORKFLOW,
        db_url=_db_url(test_db_path),
        run_id=run_id,
        config=RESPOND_CONFIG,
        timeout=30.0,
    )
    assert "SUCCESS" in result.stdout, (
        f"Normal recovery should succeed.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_orphaned_operations_are_purged_on_recovery(test_db_path: Path) -> None:
    """Recovery should purge orphaned operation_outputs beyond the journal
    boundary and complete successfully.

    The fix should detect the replay-to-fresh transition and purge all
    operation_outputs rows beyond the current fid before fresh execution begins.
    """
    run_id = "purged-orphan-test"

    max_fid = _run_interrupt_and_get_max_fid(test_db_path, run_id)

    orphan_fid = max_fid + 1
    _inject_orphaned_operation(
        test_db_path, run_id, orphan_fid, function_name="ORPHAN_FROM_CRASH"
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
    assert orphan_fn != "ORPHAN_FROM_CRASH", (
        f"Orphaned operation at fid {orphan_fid} should have been purged, "
        f"but found function_name={orphan_fn}"
    )
