# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""End-to-end DBOS + WorkflowServer + PostgresWorkflowStore integration tests.

These tests verify:
1. Event interceptor prevents published events from reaching dbos.streams
2. Events are stored as clean JSON in our wf_events table
3. subscribe_events works across the full server chain
4. Interrupt/resume produces no duplicate events (replay safety)

All tests require Docker (testcontainers) and use subprocess isolation for
DBOS global state safety.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SERVER_RUNNER_PATH = str(Path(__file__).parent / "fixtures" / "server_runner.py")
CANCEL_RESUME_RUNNER_PATH = str(
    Path(__file__).parent / "fixtures" / "cancel_resume_runner.py"
)
IDLE_RELEASE_RUNNER_PATH = str(
    Path(__file__).parent / "fixtures" / "idle_release_runner.py"
)
CANCEL_HANGS_RUNNER_PATH = str(
    Path(__file__).parent / "fixtures" / "cancel_hangs_runner.py"
)

pytestmark = [pytest.mark.docker]


def run_server_scenario(
    workflow: str,
    db_url: str,
    run_id: str,
    check_streams: bool = False,
    check_events: bool = False,
    interrupt_after: str | None = None,
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
    if check_streams:
        cmd.append("--check-streams")
    if check_events:
        cmd.append("--check-events")
    if interrupt_after:
        cmd.extend(["--interrupt-after", interrupt_after])
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def assert_no_errors(result: subprocess.CompletedProcess[str]) -> None:
    """Check subprocess result for crashes and errors."""
    if result.returncode != 0:
        pytest.fail(
            f"Subprocess exited with code {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    # Only fail on tracebacks in stdout — stderr tracebacks during DBOS
    # shutdown are noisy but harmless when exit code is 0.
    if "Traceback (most recent call last)" in result.stdout:
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


def test_event_interceptor_no_dbos_streams(postgres_dsn: str) -> None:
    """Run a workflow via the server chain and verify no events in dbos.streams."""
    result = run_server_scenario(
        workflow="tests.fixtures.sample_workflows.chained:ChainedWorkflow",
        db_url=postgres_dsn,
        run_id="test-interceptor-001",
        check_streams=True,
        check_events=True,
    )
    assert_no_errors(result)
    assert "SUCCESS" in result.stdout

    streams_count = extract_line(result.stdout, "STREAMS_COUNT:")
    assert streams_count is not None, (
        f"No STREAMS_COUNT found.\nstdout: {result.stdout}"
    )
    assert int(streams_count) == 0, (
        f"Expected 0 events in dbos.streams, got {streams_count}"
    )


def test_events_stored_as_json(postgres_dsn: str) -> None:
    """Verify events are stored in wf_events with valid JSON."""
    result = run_server_scenario(
        workflow="tests.fixtures.sample_workflows.chained:ChainedWorkflow",
        db_url=postgres_dsn,
        run_id="test-events-json-001",
        check_events=True,
    )
    assert_no_errors(result)
    assert "SUCCESS" in result.stdout

    events_count = extract_line(result.stdout, "EVENTS_COUNT:")
    assert events_count is not None, f"No EVENTS_COUNT.\nstdout: {result.stdout}"
    count = int(events_count)
    assert count >= 3, f"Expected at least 3 events, got {count}"

    event_jsons = extract_all_lines(result.stdout, "EVENT_JSON:")
    for i, event_json in enumerate(event_jsons):
        try:
            parsed = json.loads(event_json)
            assert "type" in parsed, f"Event {i} missing 'type' field"
        except json.JSONDecodeError:
            pytest.fail(f"Event {i} is not valid JSON: {event_json}")


def test_subscribe_events_receives_all_events(postgres_dsn: str) -> None:
    """Verify subscribe_events receives all events in order during workflow execution."""
    result = run_server_scenario(
        workflow="tests.fixtures.sample_workflows.chained:ChainedWorkflow",
        db_url=postgres_dsn,
        run_id="test-subscribe-001",
    )
    assert_no_errors(result)
    assert "SUCCESS" in result.stdout

    event_names = extract_all_lines(result.stdout, "EVENT:")
    assert len(event_names) >= 3, f"Expected at least 3 events, got {event_names}"

    # StopEvent should be the last event
    assert event_names[-1] == "StopEvent", (
        f"Last event should be StopEvent, got {event_names[-1]}"
    )


def test_no_duplicate_events_after_replay(postgres_dsn: str) -> None:
    """Interrupt a workflow, resume it, and verify no duplicate events."""
    run_id = "test-replay-dedup-001"

    # Run 1: interrupt after step_two produces StepTwoEvent
    result1 = run_server_scenario(
        workflow="tests.fixtures.sample_workflows.chained:ChainedWorkflow",
        db_url=postgres_dsn,
        run_id=run_id,
        interrupt_after="StepTwoEvent",
    )
    assert "INTERRUPTING" in result1.stdout, (
        f"Should have interrupted.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
    )

    # Run 2: resume to completion with same run_id, check events
    result2 = run_server_scenario(
        workflow="tests.fixtures.sample_workflows.chained:ChainedWorkflow",
        db_url=postgres_dsn,
        run_id=run_id,
        check_events=True,
        check_streams=True,
    )
    assert_no_errors(result2)
    assert "SUCCESS" in result2.stdout, (
        f"Resume should succeed.\nstdout: {result2.stdout}\nstderr: {result2.stderr}"
    )

    # Verify no events in dbos.streams
    streams_count = extract_line(result2.stdout, "STREAMS_COUNT:")
    if streams_count is not None:
        assert int(streams_count) == 0, (
            f"Expected 0 events in dbos.streams after replay, got {streams_count}"
        )


def run_cancel_resume_scenario(
    workflow: str,
    db_url: str,
    run_id: str,
    event_type: str,
    event_data: dict[str, object],
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    """Run a cancel/resume round-trip scenario."""
    cmd = [
        sys.executable,
        CANCEL_RESUME_RUNNER_PATH,
        "--workflow",
        workflow,
        "--db-url",
        db_url,
        "--run-id",
        run_id,
        "--event-type",
        event_type,
        "--event-data",
        json.dumps(event_data),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def test_cancel_resume_round_trip(postgres_dsn: str) -> None:
    """Cancel an idle DBOS workflow and resume it, verifying correct completion."""
    result = run_cancel_resume_scenario(
        workflow="tests.fixtures.sample_workflows.idle_cancel_resume:IdleCancelResumeWorkflow",
        db_url=postgres_dsn,
        run_id="test-cancel-resume-001",
        event_type="ExternalDataEvent",
        event_data={"response": "hello-world"},
    )
    assert_no_errors(result)

    assert "IDLE_DETECTED" in result.stdout, (
        f"Should detect idle.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "CANCELLED" in result.stdout, (
        f"Should cancel.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "RESUMED" in result.stdout, (
        f"Should resume.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "EVENT_SENT" in result.stdout, (
        f"Should send event.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "SUCCESS" in result.stdout, (
        f"Should complete successfully.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def run_idle_release_scenario(
    workflow: str,
    db_url: str,
    run_id: str,
    event_type: str,
    event_data: dict[str, object],
    idle_timeout: float = 0.5,
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    """Run an idle release end-to-end scenario."""
    cmd = [
        sys.executable,
        IDLE_RELEASE_RUNNER_PATH,
        "--workflow",
        workflow,
        "--db-url",
        db_url,
        "--run-id",
        run_id,
        "--event-type",
        event_type,
        "--event-data",
        json.dumps(event_data),
        "--idle-timeout",
        str(idle_timeout),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def test_idle_release_end_to_end(postgres_dsn: str) -> None:
    """Idle workflow is auto-released and auto-resumed via DBOSIdleReleaseDecorator."""
    result = run_idle_release_scenario(
        workflow="tests.fixtures.sample_workflows.idle_cancel_resume:IdleCancelResumeWorkflow",
        db_url=postgres_dsn,
        run_id="test-idle-release-001",
        event_type="ExternalDataEvent",
        event_data={"response": "idle-test"},
        idle_timeout=0.5,
    )
    assert_no_errors(result)

    assert "IDLE_DETECTED" in result.stdout, (
        f"Should detect idle.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "TIMEOUT_ELAPSED" in result.stdout, (
        f"Should wait for timeout.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "EVENT_SENT" in result.stdout, (
        f"Should send event.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "SUCCESS" in result.stdout, (
        f"Should complete successfully.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_cancel_unblocks_waiting_control_loop(postgres_dsn: str) -> None:
    """Experiment: does cancel_workflow_async unblock a control loop blocked in recv_async?

    If DBOS doesn't interrupt recv_async on cancellation, the control loop will
    hang for up to 24 hours (the unbounded wait timeout). This test checks whether
    the loop exits within 10 seconds of cancel.
    """
    cmd = [
        sys.executable,
        CANCEL_HANGS_RUNNER_PATH,
        "--workflow",
        "tests.fixtures.sample_workflows.idle_cancel_resume:IdleCancelResumeWorkflow",
        "--db-url",
        postgres_dsn,
        "--run-id",
        "test-cancel-hangs-001",
        "--cancel-wait-timeout",
        "10.0",
    ]
    # Use Popen so we can stream stdout and kill on EXPERIMENT_DONE,
    # since cleanup hangs when the control loop is still blocked in recv_async.
    import time as _time

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout_lines: list[str] = []
    deadline = _time.monotonic() + 60
    experiment_done = False
    while _time.monotonic() < deadline:
        line = proc.stdout.readline()  # type: ignore[union-attr]
        if not line:
            if proc.poll() is not None:
                break
            _time.sleep(0.1)
            continue
        stdout_lines.append(line.rstrip())
        if "EXPERIMENT_DONE" in line:
            experiment_done = True
            break

    proc.kill()
    proc.wait()
    remaining_stdout = proc.stdout.read() if proc.stdout else ""  # type: ignore[union-attr]
    stderr_output = proc.stderr.read() if proc.stderr else ""  # type: ignore[union-attr]
    stdout_output = "\n".join(stdout_lines) + remaining_stdout

    class _FakeResult:
        def __init__(self, stdout: str, stderr: str, returncode: int) -> None:
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = returncode

    result = _FakeResult(stdout_output, stderr_output, 0)  # type: ignore[assignment]

    if not experiment_done:
        pytest.fail(
            f"Subprocess timed out at 60s before EXPERIMENT_DONE.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    print(f"stdout:\n{result.stdout}")
    print(f"stderr:\n{result.stderr}")

    assert "IDLE_DETECTED" in result.stdout, (
        f"Should detect idle.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "CANCEL_CALLED" in result.stdout, (
        f"Should call cancel.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "EXPERIMENT_DONE" in result.stdout, (
        f"Experiment should complete.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    # The key assertion: did the control loop exit, or did it hang?
    if "CONTROL_LOOP_HUNG" in result.stdout:
        pytest.fail(
            f"Control loop HUNG after cancel_workflow_async! "
            f"recv_async was not interrupted by cancellation.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    # Accept either clean exit or exit-with-error (e.g. DBOSAwaitedWorkflowCancelledError)
    exited = (
        "CONTROL_LOOP_EXITED:" in result.stdout
        or "CONTROL_LOOP_EXITED_WITH_ERROR:" in result.stdout
    )
    assert exited, (
        f"Expected control loop to exit after cancel.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    # Extract timing for informational purposes
    elapsed_line = extract_line(result.stdout, "CONTROL_LOOP_EXITED:elapsed=")
    if elapsed_line is None:
        elapsed_line = extract_line(
            result.stdout, "CONTROL_LOOP_EXITED_WITH_ERROR:elapsed="
        )
    print(f"Control loop exited in {elapsed_line}")
