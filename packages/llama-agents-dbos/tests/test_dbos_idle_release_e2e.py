# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""End-to-end idle release tests over live HTTP with subprocess isolation.

Tests the full idle release → purge → resume cycle by starting a real HTTP
server inside a subprocess and exercising it via WorkflowClient. Validates
both event stream continuity (events flow across idle/resume boundary) and
handler completion status (the GET /handlers/{id} API returns "completed").

Parameterized for SQLite and PostgreSQL (Docker).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

HTTP_RUNNER_PATH = str(
    Path(__file__).parent / "fixtures" / "idle_release_http_runner.py"
)


def _run_http_idle_release(
    db_url: str, idle_timeout: float = 0.5
) -> tuple[str, str, str]:
    """Run the HTTP idle release runner and return stdout."""
    result = subprocess.run(
        [
            sys.executable,
            HTTP_RUNNER_PATH,
            "--db-url",
            db_url,
            "--idle-timeout",
            str(idle_timeout),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    stdout = result.stdout
    stderr = result.stderr
    combined = stdout + stderr
    return stdout, stderr, combined


def _assert_idle_release_http(stdout: str, stderr: str, combined: str) -> None:
    """Assert the full idle release HTTP cycle completed correctly."""
    # Stream phase: saw idle event
    assert "IDLE_DETECTED" in stdout, (
        f"Should detect idle via event stream.\nstdout: {stdout}\nstderr: {stderr}"
    )

    # Timeout elapsed and handler still running before resume
    assert "TIMEOUT_ELAPSED" in stdout, (
        f"Should wait for timeout.\nstdout: {stdout}\nstderr: {stderr}"
    )
    assert "PRE_RESUME_STATUS:running" in stdout, (
        f"Handler should still be 'running' before resume.\nstdout: {stdout}\nstderr: {stderr}"
    )

    # Event was sent successfully
    assert "SEND_STATUS:sent" in stdout, (
        f"Event send should succeed.\nstdout: {stdout}\nstderr: {stderr}"
    )

    # Resumed stream includes StopEvent (stream continuity across idle/resume)
    assert "RESUMED_STREAM:StopEvent" in stdout, (
        f"Resumed stream should include StopEvent.\nstdout: {stdout}\nstderr: {stderr}"
    )

    # Handler status API returns "completed"
    assert "FINAL_STATUS:completed" in stdout, (
        f"Handler should be marked completed.\nstdout: {stdout}\nstderr: {stderr}"
    )

    # Result is correct
    assert "RESULT:got:world" in stdout, (
        f"Result should be 'got:world'.\nstdout: {stdout}\nstderr: {stderr}"
    )

    # No DBOS determinism errors
    assert "DBOSUnexpectedStepError" not in combined, (
        f"DBOS determinism error!\nstdout: {stdout}\nstderr: {stderr}"
    )
    assert "Non-deterministic execution" not in combined, (
        f"Journal non-determinism!\nstdout: {stdout}\nstderr: {stderr}"
    )

    assert "SUCCESS" in stdout, (
        f"Should complete successfully.\nstdout: {stdout}\nstderr: {stderr}"
    )


def test_idle_release_http_sqlite(tmp_path: Path) -> None:
    """Full idle release cycle over HTTP with SQLite backend."""
    db_path = tmp_path / "idle_http_test.sqlite3"
    db_url = f"sqlite+pysqlite:///{db_path}?check_same_thread=false"
    stdout, stderr, combined = _run_http_idle_release(db_url)
    _assert_idle_release_http(stdout, stderr, combined)


@pytest.mark.docker
def test_idle_release_http_postgres(postgres_dsn: str) -> None:
    """Full idle release cycle over HTTP with PostgreSQL backend."""
    # Convert asyncpg DSN to SQLAlchemy format for DBOS
    db_url = postgres_dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    stdout, stderr, combined = _run_http_idle_release(db_url)
    _assert_idle_release_http(stdout, stderr, combined)
