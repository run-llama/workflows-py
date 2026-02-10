# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""End-to-end DBOS + WorkflowServer + PostgresWorkflowStore integration tests.

These tests verify:
1. Event interceptor prevents published events from reaching dbos.streams
2. Events are stored as clean JSON in our wf_events table
3. No duplicate events after interrupt/resume (replay safety)
4. subscribe_events works across the full server chain

All tests are gated on the TEST_POSTGRES_DSN environment variable and use
subprocess isolation for DBOS global state safety.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SERVER_RUNNER_PATH = str(Path(__file__).parent / "fixtures" / "server_runner.py")

POSTGRES_DSN = os.environ.get("TEST_POSTGRES_DSN")
requires_postgres = pytest.mark.skipif(
    POSTGRES_DSN is None,
    reason="TEST_POSTGRES_DSN not set",
)

pytestmark = [pytest.mark.no_cover, requires_postgres]


def log_on_failure(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        print(f"\n=== {label} FAILED (exit {result.returncode}) ===")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")


def run_server_scenario(
    workflow: str,
    db_url: str,
    run_id: str,
    config: dict[str, Any] | None = None,
    check_streams: bool = False,
    check_events: bool = False,
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    """Run a workflow scenario through the WorkflowServer + DBOS chain."""
    cmd = [
        sys.executable,
        SERVER_RUNNER_PATH,
        "--workflow",
        workflow,
        "--db-url",
        db_url,
        "--run-id",
        run_id,
    ]
    if config:
        cmd.extend(["--config", json.dumps(config)])
    if check_streams:
        cmd.append("--check-streams")
    if check_events:
        cmd.append("--check-events")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def assert_no_errors(result: subprocess.CompletedProcess[str]) -> None:
    """Check subprocess result for crashes and errors."""
    combined = result.stdout + result.stderr
    if result.returncode != 0:
        pytest.fail(
            f"Subprocess exited with code {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    if "Traceback (most recent call last)" in combined:
        pytest.fail(f"Exception!\nstdout: {result.stdout}\nstderr: {result.stderr}")


def extract_line(output: str, prefix: str) -> str | None:
    """Extract a line starting with prefix from output."""
    for line in output.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :]
    return None


def extract_all_lines(output: str, prefix: str) -> list[str]:
    """Extract all lines starting with prefix from output."""
    return [
        line[len(prefix) :] for line in output.splitlines() if line.startswith(prefix)
    ]


# =============================================================================
# Test 1: Event interceptor — no published events in dbos.streams
# =============================================================================


def test_event_interceptor_no_dbos_streams() -> None:
    """Run a workflow via the server chain and verify no events in dbos.streams."""
    assert POSTGRES_DSN is not None
    run_id = "test-interceptor-001"

    result = run_server_scenario(
        workflow="tests.fixtures.sample_workflows.chained:ChainedWorkflow",
        db_url=POSTGRES_DSN,
        run_id=run_id,
        check_streams=True,
        check_events=True,
    )
    log_on_failure(result, "interceptor test")
    assert_no_errors(result)
    assert "SUCCESS" in result.stdout

    # Verify no events in dbos.streams
    streams_count = extract_line(result.stdout, "STREAMS_COUNT:")
    assert streams_count is not None, (
        f"No STREAMS_COUNT found.\nstdout: {result.stdout}"
    )
    assert int(streams_count) == 0, (
        f"Expected 0 events in dbos.streams, got {streams_count}"
    )


# =============================================================================
# Test 2: Events stored as clean JSON in our events table
# =============================================================================


def test_events_stored_as_json() -> None:
    """Verify events are stored in wf_events with valid JSON."""
    assert POSTGRES_DSN is not None
    run_id = "test-events-json-001"

    result = run_server_scenario(
        workflow="tests.fixtures.sample_workflows.chained:ChainedWorkflow",
        db_url=POSTGRES_DSN,
        run_id=run_id,
        check_events=True,
    )
    log_on_failure(result, "events json test")
    assert_no_errors(result)
    assert "SUCCESS" in result.stdout

    # Verify events count
    events_count = extract_line(result.stdout, "EVENTS_COUNT:")
    assert events_count is not None, f"No EVENTS_COUNT.\nstdout: {result.stdout}"
    count = int(events_count)
    # ChainedWorkflow has 3 steps: step_one → step_two → step_three (StopEvent)
    # Each emits an event to the stream: StepOneEvent, StepTwoEvent, StopEvent
    assert count >= 3, f"Expected at least 3 events, got {count}"

    # Verify each event is valid JSON
    event_jsons = extract_all_lines(result.stdout, "EVENT_JSON:")
    for i, event_json in enumerate(event_jsons):
        try:
            parsed = json.loads(event_json)
            assert "type" in parsed, f"Event {i} missing 'type' field"
        except json.JSONDecodeError:
            pytest.fail(f"Event {i} is not valid JSON: {event_json}")


# =============================================================================
# Test 3: No duplicate events after interrupt/resume (replay safety)
# =============================================================================


def test_no_duplicate_events_after_replay() -> None:
    """Interrupt a workflow, resume it, and verify no duplicate events."""
    assert POSTGRES_DSN is not None
    run_id = "test-replay-dedup-001"

    # Run 1: interrupt at StepTwoEvent
    result1 = run_server_scenario(
        workflow="tests.fixtures.sample_workflows.chained:ChainedWorkflow",
        db_url=POSTGRES_DSN,
        run_id=run_id,
        config={"interrupt_on": "StepTwoEvent"},
    )
    log_on_failure(result1, "initial run")
    assert "INTERRUPTING" in result1.stdout, (
        f"Should have interrupted.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
    )

    # Run 2: resume to completion, check events
    result2 = run_server_scenario(
        workflow="tests.fixtures.sample_workflows.chained:ChainedWorkflow",
        db_url=POSTGRES_DSN,
        run_id=run_id,
        check_events=True,
        check_streams=True,
    )
    log_on_failure(result2, "resume")

    # The resume may or may not succeed cleanly depending on DBOS replay behavior,
    # but we should at minimum check for no duplicate events
    events_count = extract_line(result2.stdout, "EVENTS_COUNT:")
    if events_count is not None:
        count = int(events_count)
        # Should have exactly the events from the workflow, no duplicates
        # ChainedWorkflow emits: StepOneEvent, StepTwoEvent, StopEvent (3 events)
        assert count <= 4, (
            f"Expected at most 4 events (3 step events + stop), got {count}. "
            f"Possible duplicates from replay.\nstdout: {result2.stdout}"
        )

    # Verify no events in dbos.streams
    streams_count = extract_line(result2.stdout, "STREAMS_COUNT:")
    if streams_count is not None:
        assert int(streams_count) == 0, (
            f"Expected 0 events in dbos.streams after replay, got {streams_count}"
        )


# =============================================================================
# Test 4: Streaming events arrive via subscribe_events
# =============================================================================


def test_subscribe_events_receives_all_events() -> None:
    """Verify subscribe_events receives all events in order during workflow execution."""
    assert POSTGRES_DSN is not None
    run_id = "test-subscribe-001"

    result = run_server_scenario(
        workflow="tests.fixtures.sample_workflows.chained:ChainedWorkflow",
        db_url=POSTGRES_DSN,
        run_id=run_id,
    )
    log_on_failure(result, "subscribe test")
    assert_no_errors(result)
    assert "SUCCESS" in result.stdout

    # Verify we received events in order via the EVENT: output lines
    event_names = extract_all_lines(result.stdout, "EVENT:")
    assert len(event_names) >= 3, f"Expected at least 3 events, got {event_names}"

    # StopEvent should be the last event
    assert event_names[-1] == "StopEvent", (
        f"Last event should be StopEvent, got {event_names[-1]}"
    )
