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
RESTART_EXECUTOR_ID = "test-replica-restart"


def _start_idle_server(
    port: int,
    db_url: str,
    idle_timeout: float,
    executor_id: str | None = None,
) -> subprocess.Popen[str]:
    cmd = [
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
    ]
    if executor_id is not None:
        cmd.extend(["--executor-id", executor_id])
    return subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _stop_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


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
        _stop_server(proc)


async def _wait_for_handler_status(
    client: WorkflowClient, handler_id: str, status: str, timeout: float = 30.0
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        handler = await client.get_handler(handler_id)
        if handler.status == status:
            return
        await asyncio.sleep(0.25)
    raise AssertionError(f"Handler {handler_id} did not reach status {status}")


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


@pytest.mark.timeout(60)
async def test_restart_and_recover_http_workflow(tmp_path: Path) -> None:
    """Recovered startup workflow survives a restart and accepts input."""
    port = 18012
    db_path = tmp_path / "restart_recovery.sqlite3"
    db_url = f"sqlite+pysqlite:///{db_path}?check_same_thread=false"

    proc = _start_idle_server(
        port,
        db_url,
        IDLE_TIMEOUT,
        executor_id=RESTART_EXECUTOR_ID,
    )
    client = WorkflowClient(base_url=f"http://localhost:{port}")
    handler_id = ""
    run_id = ""
    try:
        _wait_for_server(proc, port)

        handler = await client.run_workflow_nowait("test")
        handler_id = handler.handler_id
        run_id = handler.run_id or ""
        assert run_id, "Workflow should have a run_id"

        stream = client.get_workflow_events(handler_id)
        async for env in stream:
            if env.type == "AskInputEvent":
                break

        last_sequence = stream.last_sequence
        await stream.aclose()

        _stop_server(proc)

        proc = _start_idle_server(
            port,
            db_url,
            IDLE_TIMEOUT,
            executor_id=RESTART_EXECUTOR_ID,
        )
        _wait_for_server(proc, port)

        recovered = await client.get_handler(handler_id)
        assert recovered.status == "running"
        assert recovered.run_id == run_id

        send_resp = await client.send_event(handler_id, UserInput(response="world"))
        assert send_resp.status == "sent"

        await _wait_for_handler_status(client, handler_id, "completed")
        completed = await client.get_handler(handler_id)
        assert completed.result is not None
        assert completed.result.value.get("result", {}).get("response") == "world"

        resumed_stream = client.get_workflow_events(
            handler_id, after_sequence=last_sequence
        )
        saw_stop = False
        async for env in resumed_stream:
            if env.type == "StopEvent":
                saw_stop = True
                break
        await resumed_stream.aclose()
        assert saw_stop, "Recovered workflow should emit StopEvent after restart"
    finally:
        _stop_server(proc)
