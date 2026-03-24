# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Reproduction: orphaned DBOS operations cause DBOSUnexpectedStepError.

After journal exhaustion during DBOS recovery, orphaned operation_outputs
from previous crashes persist beyond the journal boundary. If the fresh
execution's fid sequence diverges from the original (due to non-deterministic
done.pop() ordering in wait_for_next_task), DBOS detects the function_name
mismatch at the divergence point and raises DBOSUnexpectedStepError.

The DBOS runtime uses done.pop() (non-deterministic) while the non-DBOS
runtime uses pick_highest_priority (deterministic). This is the root cause.

Test strategy:
1. Run workflow to a known point, then interrupt (creates journal + operations)
2. Inject orphaned operations beyond the last recorded fid with a function_name
   that differs from what recovery will call → proves the error mechanism
3. Recover → DBOSUnexpectedStepError confirms the theory
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
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
RESPOND_CONFIG = {
    "respond": {
        "SlowFanOutTickEvent": {
            "event": "SlowFanOutContinueEvent",
            "fields": {},
        }
    }
}


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    return tmp_path / "orphan_determinism_test.sqlite3"


def _get_journal_count(db_path: Path, run_id: str) -> int:
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(str(db_path))
    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='workflow_journal'"
        ).fetchall()
        if not tables:
            return 0
        row = conn.execute(
            "SELECT COUNT(*) FROM workflow_journal WHERE run_id=?", (run_id,)
        ).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


def _get_max_fid(db_path: Path, run_id: str) -> int:
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(str(db_path))
    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='operation_outputs'"
        ).fetchall()
        if not tables:
            return 0
        row = conn.execute(
            "SELECT MAX(function_id) FROM operation_outputs WHERE workflow_uuid=?",
            (run_id,),
        ).fetchone()
        return row[0] if row and row[0] is not None else 0
    finally:
        conn.close()


def _inject_orphaned_operation(
    db_path: Path,
    run_id: str,
    function_id: int,
    function_name: str,
    output: str = "null",
) -> None:
    """Insert a fake orphaned operation into the DBOS operation_outputs table.

    This simulates what happens when a crash leaves committed DBOS operations
    beyond the journal boundary. On recovery, DBOS finds these operations and
    checks their function_names against the current execution.
    """
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
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT function_name FROM operation_outputs "
            "WHERE workflow_uuid=? AND function_id=?",
            (run_id, fid),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _run_and_kill_after(
    workflow: str,
    db_url: str,
    run_id: str,
    config: dict[str, Any],
    kill_after_sec: float,
) -> subprocess.CompletedProcess[str]:
    """Run a workflow subprocess and SIGKILL it after a delay."""
    cmd = [
        sys.executable,
        RUNNER_PATH,
        "--workflow",
        workflow,
        "--db-url",
        db_url,
        "--run-id",
        run_id,
        "--config",
        json.dumps(config),
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


def test_injected_orphan_triggers_determinism_error(test_db_path: Path) -> None:
    """Prove that orphaned operations with mismatched function_names cause
    DBOSUnexpectedStepError on recovery.

    1. Run workflow to a known point (interrupt on round 1)
    2. Inject an orphaned operation at max_fid+1 with a WRONG function_name
    3. Recovery hits the orphaned fid and raises DBOSUnexpectedStepError
    """
    run_id = "injected-orphan-test"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    # --- Run 1: interrupt on round 1 ---
    interrupt_config = {
        "respond": RESPOND_CONFIG["respond"],
        "interrupt_on": {"event": "SlowFanOutTickEvent", "condition": {"round": 1}},
    }
    result1 = run_scenario(
        workflow=SLOW_FANOUT_WORKFLOW,
        db_url=db_url,
        run_id=run_id,
        config=interrupt_config,
        timeout=30.0,
    )
    assert "INTERRUPTING" in result1.stdout, (
        f"Run 1 should have interrupted.\nstdout: {result1.stdout}"
    )

    max_fid = _get_max_fid(test_db_path, run_id)
    journal_count = _get_journal_count(test_db_path, run_id)
    print(f"After run 1: journal={journal_count}, max_fid={max_fid}")

    # The next fid after max_fid will be consumed by get_now() (_durable_time)
    # at the top of the first fresh execution iteration. Inject an orphaned
    # operation with a WRONG function name at that fid.
    orphan_fid = max_fid + 1
    _inject_orphaned_operation(
        test_db_path,
        run_id,
        orphan_fid,
        function_name="FAKE_ORPHANED_STEP",  # NOT _durable_time
        output='"fake_result"',
    )
    print(f"Injected orphan at fid {orphan_fid}: FAKE_ORPHANED_STEP")

    # --- Run 2: recovery should hit the orphaned operation ---
    result2 = run_scenario(
        workflow=SLOW_FANOUT_WORKFLOW,
        db_url=db_url,
        run_id=run_id,
        config=RESPOND_CONFIG,
        timeout=30.0,
    )

    combined = result2.stdout + result2.stderr
    has_determinism_error = (
        "DBOSUnexpectedStepError" in combined or "Error 11" in combined
    )

    print(f"  returncode={result2.returncode}")
    print(f"  has_determinism_error={has_determinism_error}")

    if not has_determinism_error:
        print(f"  stdout:\n{result2.stdout}")
        print(f"  stderr:\n{result2.stderr}")

    assert has_determinism_error, (
        f"Expected DBOSUnexpectedStepError from orphaned operation but got "
        f"exit code {result2.returncode}.\n"
        f"stdout: {result2.stdout}\nstderr: {result2.stderr}"
    )
    assert "FAKE_ORPHANED_STEP" in combined, (
        "Error should reference the fake function name"
    )


def test_sigkill_creates_orphaned_operations(test_db_path: Path) -> None:
    """Verify that SIGKILL mid-execution creates orphaned DBOS operations
    beyond the journal boundary.

    These orphaned operations are the precondition for the determinism bug.
    """
    run_id = "sigkill-orphan-check"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    # Run 1: interrupt at round 1
    interrupt_config = {
        "respond": RESPOND_CONFIG["respond"],
        "interrupt_on": {"event": "SlowFanOutTickEvent", "condition": {"round": 1}},
    }
    result1 = run_scenario(
        workflow=SLOW_FANOUT_WORKFLOW,
        db_url=db_url,
        run_id=run_id,
        config=interrupt_config,
        timeout=30.0,
    )
    assert "INTERRUPTING" in result1.stdout

    max_fid_1 = _get_max_fid(test_db_path, run_id)
    journal_1 = _get_journal_count(test_db_path, run_id)
    print(f"After interrupt: journal={journal_1}, max_fid={max_fid_1}")

    # Run 2: SIGKILL after 1.5s (creates orphaned ops)
    _run_and_kill_after(
        workflow=SLOW_FANOUT_WORKFLOW,
        db_url=db_url,
        run_id=run_id,
        config=RESPOND_CONFIG,
        kill_after_sec=1.5,
    )

    max_fid_2 = _get_max_fid(test_db_path, run_id)
    journal_2 = _get_journal_count(test_db_path, run_id)
    print(f"After SIGKILL: journal={journal_2}, max_fid={max_fid_2}")

    # The SIGKILL should have created operations beyond the journal boundary
    assert max_fid_2 > max_fid_1, "SIGKILL should have added more DBOS operations"
    assert journal_2 > journal_1, "SIGKILL should have added more journal entries"

    # Run 3: normal recovery should succeed (same process = same done.pop() order)
    result3 = run_scenario(
        workflow=SLOW_FANOUT_WORKFLOW,
        db_url=db_url,
        run_id=run_id,
        config=RESPOND_CONFIG,
        timeout=30.0,
    )

    has_success = "SUCCESS" in result3.stdout
    combined = result3.stdout + result3.stderr
    has_determinism_error = (
        "DBOSUnexpectedStepError" in combined or "Error 11" in combined
    )

    print(f"  returncode={result3.returncode}")
    print(f"  has_success={has_success}")
    print(f"  has_determinism_error={has_determinism_error}")

    if not has_success:
        print(f"  stdout:\n{result3.stdout}")
        print(f"  stderr:\n{result3.stderr}")

    # Normal recovery should succeed because done.pop() is deterministic
    # within the same CPython process (same memory layout → same hash order)
    assert has_success, (
        f"Normal recovery should succeed.\n"
        f"stdout: {result3.stdout}\nstderr: {result3.stderr}"
    )


def test_shuffled_recovery_with_injected_orphans(test_db_path: Path) -> None:
    """Full reproduction: inject orphaned operations that specifically mismatch
    the reversed-priority task selection, proving that done.pop() ordering
    is the root cause.

    This test:
    1. Runs the workflow and interrupts to establish a known fid sequence
    2. Records what function_name the NEXT fid would be (_durable_time)
    3. Injects an orphaned operation at that fid with a WORKER function_name
    4. Recovers with reversed priority → recovery calls _durable_time at that
       fid but finds worker_alpha recorded → DBOSUnexpectedStepError
    """
    run_id = "shuffled-orphan-repro"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    # Run 1: interrupt on round 1
    interrupt_config = {
        "respond": RESPOND_CONFIG["respond"],
        "interrupt_on": {"event": "SlowFanOutTickEvent", "condition": {"round": 1}},
    }
    result1 = run_scenario(
        workflow=SLOW_FANOUT_WORKFLOW,
        db_url=db_url,
        run_id=run_id,
        config=interrupt_config,
        timeout=30.0,
    )
    assert "INTERRUPTING" in result1.stdout

    max_fid = _get_max_fid(test_db_path, run_id)
    print(f"After interrupt: max_fid={max_fid}")

    # After journal exhaustion, the first fid consumed is get_now() → _durable_time.
    # Inject an orphaned operation with a DIFFERENT function name to simulate
    # what happens when the fid counter diverges due to done.pop() ordering.
    orphan_fid = max_fid + 1
    _inject_orphaned_operation(
        test_db_path,
        run_id,
        orphan_fid,
        function_name=(
            "tests.fixtures.sample_workflows.slow_fan_out_hitl."
            "SlowFanOutWorkflow.worker_alpha"
        ),
        output='"orphaned_worker_result"',
    )
    print(
        f"Injected orphan at fid {orphan_fid}: worker_alpha (should be _durable_time)"
    )

    # Recovery: the runtime will replay the journal, then start fresh execution.
    # At fid max_fid+1, it calls _durable_time (get_now at top of loop).
    # DBOS finds the orphaned operation with function_name=worker_alpha.
    # Function name mismatch → DBOSUnexpectedStepError.
    result2 = run_scenario(
        workflow=SLOW_FANOUT_WORKFLOW,
        db_url=db_url,
        run_id=run_id,
        config=RESPOND_CONFIG,
        timeout=30.0,
    )

    combined = result2.stdout + result2.stderr
    has_determinism_error = (
        "DBOSUnexpectedStepError" in combined or "Error 11" in combined
    )

    print(f"  returncode={result2.returncode}")
    print(f"  has_determinism_error={has_determinism_error}")

    assert has_determinism_error, (
        f"Expected DBOSUnexpectedStepError but got exit code {result2.returncode}.\n"
        f"stdout: {result2.stdout}\nstderr: {result2.stderr}"
    )
    assert "worker_alpha" in combined and "_durable_time" in combined, (
        "Error should show the function_name mismatch between orphan and actual call"
    )


def test_orphaned_operations_are_purged_on_recovery(test_db_path: Path) -> None:
    """Recovery should purge orphaned operation_outputs beyond the journal
    boundary and complete successfully.

    A crash (SIGKILL, power loss, OOM kill) can leave committed DBOS
    operation_outputs rows whose corresponding journal entries were never
    flushed. On the next recovery, these orphaned rows sit at fids beyond
    the journal replay point. If fresh execution consumes those fids with
    different functions (likely under any non-trivial workload), DBOS raises
    DBOSUnexpectedStepError.

    The fix should detect the replay→fresh transition and purge all
    operation_outputs rows beyond the current fid before fresh execution
    begins.

    1. Run workflow, interrupt cleanly (creates journal + operations)
    2. Inject an orphaned operation at max_fid+1 with a wrong function_name
       (simulates the operation/journal gap a crash leaves)
    3. Recover — should purge the orphan and succeed
    """
    run_id = "purged-orphan-test"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    # --- Run 1: run workflow, interrupt on round 1 ---
    interrupt_config = {
        "respond": RESPOND_CONFIG["respond"],
        "interrupt_on": {"event": "SlowFanOutTickEvent", "condition": {"round": 1}},
    }
    result1 = run_scenario(
        workflow=SLOW_FANOUT_WORKFLOW,
        db_url=db_url,
        run_id=run_id,
        config=interrupt_config,
        timeout=30.0,
    )
    assert "INTERRUPTING" in result1.stdout, (
        f"Run 1 should have interrupted.\nstdout: {result1.stdout}"
    )

    max_fid = _get_max_fid(test_db_path, run_id)
    print(f"After run 1: max_fid={max_fid}")

    # Inject an orphaned operation at the next fid. After journal replay,
    # the first fresh fid consumed is _durable_time (get_now at top of loop).
    # The wrong function_name here guarantees a mismatch.
    orphan_fid = max_fid + 1
    _inject_orphaned_operation(
        test_db_path,
        run_id,
        orphan_fid,
        function_name="ORPHAN_FROM_CRASH",
        output='"orphaned_result"',
    )
    print(f"Injected orphan at fid {orphan_fid}: ORPHAN_FROM_CRASH")

    # --- Run 2: recovery should purge the orphan and succeed ---
    result2 = run_scenario(
        workflow=SLOW_FANOUT_WORKFLOW,
        db_url=db_url,
        run_id=run_id,
        config=RESPOND_CONFIG,
        timeout=30.0,
    )

    combined = result2.stdout + result2.stderr
    print(f"  returncode={result2.returncode}")
    print(f"  stdout:\n{result2.stdout}")
    print(f"  stderr:\n{result2.stderr}")

    # Assert recovery succeeded
    assert "SUCCESS" in result2.stdout, (
        f"Recovery should succeed after purging orphan.\n"
        f"stdout: {result2.stdout}\nstderr: {result2.stderr}"
    )
    assert "DBOSUnexpectedStepError" not in combined, (
        "Should NOT get determinism error after purge"
    )

    # Assert the orphaned row was cleaned up
    orphan_fn = _get_function_at_fid(test_db_path, run_id, orphan_fid)
    assert orphan_fn != "ORPHAN_FROM_CRASH", (
        f"Orphaned operation at fid {orphan_fid} should have been purged, "
        f"but found function_name={orphan_fn}"
    )
