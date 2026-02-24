# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""End-to-end idle release tests over live HTTP with subprocess isolation.

Tests the full idle release → purge → resume cycle by starting a real HTTP
server (replica_server.py with --idle-timeout) and exercising it via
WorkflowClient. Validates event stream continuity across the idle/resume
boundary and handler completion.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from llama_agents.client import WorkflowClient
from tests.fixtures.sample_workflows.hitl import UserInput
from workflows.events import WorkflowIdleEvent

REPLICA_SERVER_PATH = str(Path(__file__).parent / "fixtures" / "replica_server.py")
WORKFLOW_PATH = "tests.fixtures.sample_workflows.hitl:TestWorkflow"
IDLE_TIMEOUT = 0.5


def _start_idle_server(
    port: int, db_url: str, idle_timeout: float
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            REPLICA_SERVER_PATH,
            "--workflow",
            WORKFLOW_PATH,
            "--db-url",
            db_url,
            "--port",
            str(port),
            "--idle-timeout",
            str(idle_timeout),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _wait_for_server(
    proc: subprocess.Popen[str], port: int, timeout: float = 30.0
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stdout = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(
                f"Server on port {port} exited with code {proc.returncode}\n"
                f"output: {stdout}"
            )
        try:
            resp = httpx.get(f"http://localhost:{port}/workflows", timeout=2.0)
            if resp.status_code == 200:
                return
        except httpx.ConnectError:
            pass
        time.sleep(0.5)
    proc.kill()
    stdout = proc.stdout.read() if proc.stdout else ""
    raise RuntimeError(
        f"Server on port {port} did not start in {timeout}s\noutput: {stdout}"
    )


async def _run_idle_release_test(port: int, db_url: str) -> None:
    """Core test logic shared between SQLite and Postgres variants."""
    proc = _start_idle_server(port, db_url, IDLE_TIMEOUT)
    try:
        _wait_for_server(proc, port)
        client = WorkflowClient(base_url=f"http://localhost:{port}")

        # 1. Start workflow
        handler = await client.run_workflow_nowait("test")
        handler_id = handler.handler_id

        # 2. Stream events until WorkflowIdleEvent
        stream = client.get_workflow_events(handler_id, include_internal_events=True)
        async for env in stream:
            event = env.load_event([WorkflowIdleEvent])
            if isinstance(event, WorkflowIdleEvent):
                break

        last_seq = stream.last_sequence

        # 3. Wait for idle timeout to elapse (release happens in background)
        await asyncio.sleep(IDLE_TIMEOUT + 1.5)

        # 4. Handler should still be "running" (released but not completed)
        h = await client.get_handler(handler_id)
        assert h.status == "running", f"Expected 'running', got '{h.status}'"

        # 5. Send event to trigger resume
        send_resp = await client.send_event(handler_id, UserInput(response="world"))
        assert send_resp.status == "sent"

        # 6. Stream events after resume, expect StopEvent
        got_stop = False
        async for env in client.get_workflow_events(
            handler_id, after_sequence=last_seq
        ):
            if env.type == "StopEvent":
                got_stop = True
                break
        assert got_stop, "Should see StopEvent after resume"

        # 7. Poll for handler completion
        for _ in range(40):
            h = await client.get_handler(handler_id)
            if h.status == "completed":
                break
            await asyncio.sleep(0.25)
        assert h.status == "completed", f"Expected 'completed', got '{h.status}'"
        assert h.result is not None
        assert h.result.value.get("result", {}).get("response") == "world"
    finally:
        proc.kill()
        proc.wait()


@pytest.mark.timeout(45)
async def test_idle_release_e2e_sqlite(tmp_path: Path) -> None:
    """Full idle release cycle over HTTP with SQLite backend."""
    db_path = tmp_path / "idle_e2e.sqlite3"
    db_url = f"sqlite+pysqlite:///{db_path}?check_same_thread=false"
    await _run_idle_release_test(18010, db_url)


@pytest.mark.docker
@pytest.mark.timeout(45)
async def test_idle_release_e2e_postgres(postgres_dsn: str) -> None:
    """Full idle release cycle over HTTP with PostgreSQL backend."""
    db_url = postgres_dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    await _run_idle_release_test(18011, db_url)
