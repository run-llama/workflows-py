# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Cross-process integration tests for DBOS distributed model.

Tests verify that two WorkflowServer replicas sharing a Postgres database
can coordinate workflow execution: one replica runs the workflow while the
other sends events and streams results.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from llama_agents.client import WorkflowClient

REPLICA_SERVER_PATH = str(Path(__file__).parent / "fixtures" / "replica_server.py")
WORKFLOW_PATH = "tests.fixtures.sample_workflows.hitl:TestWorkflow"

pytestmark = [pytest.mark.docker]

PORT_A = 18001
PORT_B = 18002


def start_replica(port: int, db_url: str) -> subprocess.Popen[str]:
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
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def wait_for_server(
    proc: subprocess.Popen[str], port: int, timeout: float = 30.0
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # Check if process died
        if proc.poll() is not None:
            stdout = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(
                f"Replica on port {port} exited with code {proc.returncode}\n"
                f"output: {stdout}"
            )
        try:
            resp = httpx.get(f"http://localhost:{port}/workflows", timeout=2.0)
            if resp.status_code == 200:
                return
        except httpx.ConnectError:
            pass
        time.sleep(0.5)
    # Grab whatever output we can
    proc.kill()
    stdout = proc.stdout.read() if proc.stdout else ""
    raise RuntimeError(
        f"Server on port {port} did not start in {timeout}s\noutput: {stdout}"
    )


@pytest.mark.timeout(60)
async def test_cross_process_event_delivery(postgres_dsn: str) -> None:
    """Event sent via Replica B reaches workflow running in Replica A."""
    from tests.fixtures.sample_workflows.hitl import UserInput

    replicas: list[subprocess.Popen[str]] = []
    try:
        # Start replicas sequentially to avoid DBOS CREATE SCHEMA race (upstream bug)
        replicas.append(start_replica(PORT_A, postgres_dsn))
        wait_for_server(replicas[0], PORT_A)
        replicas.append(start_replica(PORT_B, postgres_dsn))
        wait_for_server(replicas[1], PORT_B)

        client_a = WorkflowClient(base_url=f"http://localhost:{PORT_A}")
        client_b = WorkflowClient(base_url=f"http://localhost:{PORT_B}")

        # Start workflow on Replica A — step "ask" returns AskInputEvent
        handler = await client_a.run_workflow_nowait("test")
        handler_id = handler.handler_id

        # Send UserInput via Replica B — step "process" on Replica A receives it
        await client_b.send_event(handler_id, UserInput(response="cross-process"))

        # Stream events from Replica A — should see StopEvent
        got_stop = False
        async for event in client_a.get_workflow_events(handler_id, after_sequence=-1):
            if event.type == "StopEvent":
                got_stop = True

        assert got_stop, "Workflow should have completed with StopEvent"

        # Stream events from Replica B — should also see the same events
        got_stop_b = False
        async for event in client_b.get_workflow_events(handler_id, after_sequence=-1):
            if event.type == "StopEvent":
                got_stop_b = True

        assert got_stop_b, (
            "Replica B should also see StopEvent via shared Postgres store"
        )
    finally:
        for proc in replicas:
            proc.kill()
        for proc in replicas:
            proc.wait()
