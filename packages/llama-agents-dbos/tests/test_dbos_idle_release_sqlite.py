# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""SQLite-based idle release integration tests using subprocess isolation.

Tests the full idle release → purge → resume cycle with SQLite, verifying
that DBOS operation_outputs and our journal table are properly purged so
the resumed workflow completes without determinism errors.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

IDLE_RELEASE_RUNNER_PATH = str(
    Path(__file__).parent / "fixtures" / "idle_release_runner.py"
)


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    return tmp_path / "idle_release_test.sqlite3"


def test_idle_release_resume_completes_sqlite(test_db_path: Path) -> None:
    """Full idle release cycle: start → idle → release → purge → resume → complete."""
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    result = subprocess.run(
        [
            sys.executable,
            IDLE_RELEASE_RUNNER_PATH,
            "--workflow",
            "tests.fixtures.sample_workflows.idle_cancel_resume:IdleCancelResumeWorkflow",
            "--db-url",
            db_url,
            "--run-id",
            "test-idle-sqlite-001",
            "--event-type",
            "ExternalDataEvent",
            "--event-data",
            '{"response": "hello"}',
            "--idle-timeout",
            "0.5",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )

    stdout = result.stdout
    stderr = result.stderr
    combined = stdout + stderr

    assert "IDLE_DETECTED" in stdout, (
        f"Should detect idle.\nstdout: {stdout}\nstderr: {stderr}"
    )
    assert "TIMEOUT_ELAPSED" in stdout, (
        f"Should wait for timeout.\nstdout: {stdout}\nstderr: {stderr}"
    )
    assert "EVENT_SENT" in stdout, (
        f"Should send event.\nstdout: {stdout}\nstderr: {stderr}"
    )

    # Check no DBOS determinism errors
    assert "DBOSUnexpectedStepError" not in combined, (
        f"DBOS determinism error!\nstdout: {stdout}\nstderr: {stderr}"
    )
    assert "Non-deterministic execution" not in combined, (
        f"Journal non-determinism!\nstdout: {stdout}\nstderr: {stderr}"
    )

    assert "SUCCESS" in stdout, (
        f"Workflow should complete after resume.\nstdout: {stdout}\nstderr: {stderr}"
    )

    # Handler status should be marked "completed" (not stuck at 202)
    assert "HANDLER_COMPLETED" in stdout, (
        f"Handler should be marked completed.\nstdout: {stdout}\nstderr: {stderr}"
    )
