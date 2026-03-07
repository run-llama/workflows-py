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

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

RUNNER_PATH = str(Path(__file__).parent / "fixtures" / "runner.py")
SIMPLE_COUNTER_RUNNER_PATH = str(
    Path(__file__).parent / "fixtures" / "simple_counter_runner.py"
)


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    return tmp_path / "double_restart_test.sqlite3"


def run_scenario(
    workflow: str,
    db_url: str,
    run_id: str,
    config: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        RUNNER_PATH,
        "--workflow",
        workflow,
        "--db-url",
        db_url,
        "--run-id",
        run_id,
    ]
    if config:
        cmd.extend(["--config", json.dumps(config)])
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        pytest.fail(
            f"Subprocess timed out after {timeout}s — likely the double-restart "
            f"hang bug.\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
        raise AssertionError("unreachable")  # noqa: B904


def run_simple_counter(
    db_url: str,
    run_id: str,
    interrupt_at: int | None = None,
    target: int = 20,
    fast_polling: bool = True,
    graceful_interrupt: bool = False,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        SIMPLE_COUNTER_RUNNER_PATH,
        "--db-url",
        db_url,
        "--run-id",
        run_id,
        "--target",
        str(target),
    ]
    if interrupt_at is not None:
        cmd.extend(["--interrupt-at", str(interrupt_at)])
    if fast_polling:
        cmd.append("--fast-polling")
    if graceful_interrupt:
        cmd.append("--graceful-interrupt")
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        pytest.fail(
            f"Subprocess timed out after {timeout}s — likely the double-restart "
            f"hang bug.\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
        raise AssertionError("unreachable")  # noqa: B904


def get_journal_rows(db_path: Path, run_id: str) -> list[tuple[Any, ...]]:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT * FROM workflow_journal WHERE run_id=? ORDER BY seq_num",
            (run_id,),
        ).fetchall()
    finally:
        conn.close()


COUNTER_WORKFLOW = "tests.fixtures.sample_workflows.counter:CounterWorkflow"
RESPOND_CONFIG = {
    "respond": {
        "CounterTickEvent": {
            "event": "CounterContinueEvent",
            "fields": {},
        }
    }
}


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

    rows1 = get_journal_rows(test_db_path, run_id)
    pull_count1 = sum(1 for r in rows1 if "__pull__" in r[3])
    print(f"\nAfter run 1: {len(rows1)} journal rows ({pull_count1} __pull__)")

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

    rows2 = get_journal_rows(test_db_path, run_id)
    pull_count2 = sum(1 for r in rows2 if "__pull__" in r[3])
    print(f"After run 2: {len(rows2)} journal rows ({pull_count2} __pull__)")

    # --- Run 3: resume, should complete to 40 ---
    result3 = run_scenario(
        workflow=COUNTER_WORKFLOW,
        db_url=db_url,
        run_id=run_id,
        config=RESPOND_CONFIG,
        timeout=15.0,
    )

    rows3 = get_journal_rows(test_db_path, run_id)
    pull_count3 = sum(1 for r in rows3 if "__pull__" in r[3])
    print(f"After run 3: {len(rows3)} journal rows ({pull_count3} __pull__)")

    combined = result3.stdout + result3.stderr
    assert "DBOSUnexpectedStepError" not in combined, (
        f"DBOS determinism error!\nstdout: {result3.stdout}\nstderr: {result3.stderr}"
    )
    assert result3.returncode == 0, (
        f"Process exited with code {result3.returncode}.\n"
        f"stdout: {result3.stdout}\nstderr: {result3.stderr}"
    )
    assert "SUCCESS" in result3.stdout, (
        f"Should complete successfully after second resume.\n"
        f"stdout: {result3.stdout}\nstderr: {result3.stderr}"
    )


def test_double_restart_simple_counter_fast_polling(test_db_path: Path) -> None:
    """Two interrupt-restart cycles on the non-HITL quickstart counter.

    This matches the exact workflow from the DBOS quickstart docs.
    Uses fast polling (test default) — no HITL, plain Tick events.
    """
    run_id = "simple-counter-fast"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    # --- Run 1: start, interrupt at tick 5 ---
    result1 = run_simple_counter(
        db_url=db_url, run_id=run_id, interrupt_at=5, fast_polling=True
    )
    assert "STEP:start:complete" in result1.stdout, (
        f"Start step should complete.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
    )
    assert "INTERRUPTING" in result1.stdout, (
        f"Should interrupt at tick 5.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
    )

    rows1 = get_journal_rows(test_db_path, run_id)
    pull_count1 = sum(1 for r in rows1 if "__pull__" in r[3])
    print(f"\nAfter run 1: {len(rows1)} journal rows ({pull_count1} __pull__)")

    # --- Run 2: resume, interrupt at tick 12 ---
    result2 = run_simple_counter(
        db_url=db_url, run_id=run_id, interrupt_at=12, fast_polling=True
    )
    assert "INTERRUPTING" in result2.stdout, (
        f"Should interrupt at tick 12.\nstdout: {result2.stdout}\nstderr: {result2.stderr}"
    )

    rows2 = get_journal_rows(test_db_path, run_id)
    pull_count2 = sum(1 for r in rows2 if "__pull__" in r[3])
    print(f"After run 2: {len(rows2)} journal rows ({pull_count2} __pull__)")

    # --- Run 3: resume, should complete to 20 ---
    result3 = run_simple_counter(
        db_url=db_url, run_id=run_id, fast_polling=True, timeout=30.0
    )

    rows3 = get_journal_rows(test_db_path, run_id)
    pull_count3 = sum(1 for r in rows3 if "__pull__" in r[3])
    print(f"After run 3: {len(rows3)} journal rows ({pull_count3} __pull__)")

    combined = result3.stdout + result3.stderr
    assert "DBOSUnexpectedStepError" not in combined, (
        f"DBOS determinism error!\nstdout: {result3.stdout}\nstderr: {result3.stderr}"
    )
    assert result3.returncode == 0, (
        f"Process exited with code {result3.returncode}.\n"
        f"stdout: {result3.stdout}\nstderr: {result3.stderr}"
    )
    assert "SUCCESS" in result3.stdout, (
        f"Should complete successfully.\n"
        f"stdout: {result3.stdout}\nstderr: {result3.stderr}"
    )


def test_double_restart_graceful_shutdown(test_db_path: Path) -> None:
    """Two graceful-shutdown restart cycles on the simple counter.

    This is the actual bug scenario: when the process shuts down gracefully
    (e.g. Ctrl+C caught by try/finally), runtime.destroy() calls close()
    on the adapter. The old close() implementation sent a _DBOSInternalShutdown
    message via DBOS.send(), which persisted to the notifications table.

    On resume, DBOS replays that notification, which poisons the adapter's
    _closed flag. After two such cycles, the accumulated __pull__ journal
    entries cause DBOS's function_id tracking to diverge, and recv_async
    blocks forever.

    Hard-kill tests (os._exit) don't catch this because close() never runs.
    """
    run_id = "graceful-double-restart"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    # --- Run 1: start, graceful interrupt at tick 5 ---
    result1 = run_simple_counter(
        db_url=db_url,
        run_id=run_id,
        interrupt_at=5,
        fast_polling=True,
        graceful_interrupt=True,
    )
    assert "STEP:start:complete" in result1.stdout, (
        f"Start step should complete.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
    )
    assert "INTERRUPTING" in result1.stdout, (
        f"Should interrupt at tick 5.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
    )
    assert "GRACEFUL_SHUTDOWN" in result1.stdout, (
        f"Should show graceful shutdown.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
    )

    # --- Run 2: resume, graceful interrupt at tick 12 ---
    result2 = run_simple_counter(
        db_url=db_url,
        run_id=run_id,
        interrupt_at=12,
        fast_polling=True,
        graceful_interrupt=True,
    )
    assert "INTERRUPTING" in result2.stdout, (
        f"Should interrupt at tick 12.\nstdout: {result2.stdout}\nstderr: {result2.stderr}"
    )
    assert "GRACEFUL_SHUTDOWN" in result2.stdout, (
        f"Should show graceful shutdown.\nstdout: {result2.stdout}\nstderr: {result2.stderr}"
    )

    # --- Run 3: resume, should complete to 20 ---
    result3 = run_simple_counter(
        db_url=db_url, run_id=run_id, fast_polling=True, timeout=30.0
    )

    combined = result3.stdout + result3.stderr
    assert "DBOSUnexpectedStepError" not in combined, (
        f"DBOS determinism error!\nstdout: {result3.stdout}\nstderr: {result3.stderr}"
    )
    assert result3.returncode == 0, (
        f"Process exited with code {result3.returncode}.\n"
        f"stdout: {result3.stdout}\nstderr: {result3.stderr}"
    )
    assert "SUCCESS" in result3.stdout, (
        f"Should complete after two graceful shutdowns.\n"
        f"stdout: {result3.stdout}\nstderr: {result3.stderr}"
    )


def test_double_restart_simple_counter_default_polling(test_db_path: Path) -> None:
    """Same as above but with DEFAULT DBOS polling intervals.

    The user's quickstart example uses DBOSRuntime() with no polling config,
    meaning DBOS uses its default notification_listener_polling_interval_sec.
    This may behave differently during recovery than the fast polling variant.
    """
    run_id = "simple-counter-default"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    # --- Run 1: start, interrupt at tick 5 ---
    result1 = run_simple_counter(
        db_url=db_url, run_id=run_id, interrupt_at=5, fast_polling=False
    )
    assert "STEP:start:complete" in result1.stdout, (
        f"Start step should complete.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
    )
    assert "INTERRUPTING" in result1.stdout, (
        f"Should interrupt at tick 5.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
    )

    rows1 = get_journal_rows(test_db_path, run_id)
    pull_count1 = sum(1 for r in rows1 if "__pull__" in r[3])
    print(f"\nAfter run 1: {len(rows1)} journal rows ({pull_count1} __pull__)")

    # --- Run 2: resume, interrupt at tick 12 ---
    result2 = run_simple_counter(
        db_url=db_url, run_id=run_id, interrupt_at=12, fast_polling=False
    )
    assert "INTERRUPTING" in result2.stdout, (
        f"Should interrupt at tick 12.\nstdout: {result2.stdout}\nstderr: {result2.stderr}"
    )

    rows2 = get_journal_rows(test_db_path, run_id)
    pull_count2 = sum(1 for r in rows2 if "__pull__" in r[3])
    print(f"After run 2: {len(rows2)} journal rows ({pull_count2} __pull__)")

    # --- Run 3: resume, should complete to 20 ---
    result3 = run_simple_counter(
        db_url=db_url, run_id=run_id, fast_polling=False, timeout=30.0
    )

    rows3 = get_journal_rows(test_db_path, run_id)
    pull_count3 = sum(1 for r in rows3 if "__pull__" in r[3])
    print(f"After run 3: {len(rows3)} journal rows ({pull_count3} __pull__)")

    combined = result3.stdout + result3.stderr
    assert "DBOSUnexpectedStepError" not in combined, (
        f"DBOS determinism error!\nstdout: {result3.stdout}\nstderr: {result3.stderr}"
    )
    assert result3.returncode == 0, (
        f"Process exited with code {result3.returncode}.\n"
        f"stdout: {result3.stdout}\nstderr: {result3.stderr}"
    )
    assert "SUCCESS" in result3.stdout, (
        f"Should complete successfully.\n"
        f"stdout: {result3.stdout}\nstderr: {result3.stderr}"
    )
