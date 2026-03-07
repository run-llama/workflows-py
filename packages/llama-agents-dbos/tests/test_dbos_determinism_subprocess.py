# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Test DBOS determinism with subprocess isolation and real interruption.

This test spawns subprocesses to properly isolate DBOS state and simulate
real Ctrl+C interruptions during workflow execution.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import (
    assert_no_determinism_errors,  # pyright: ignore[reportAttributeAccessIssue]
    log_on_failure,  # pyright: ignore[reportAttributeAccessIssue]
    run_scenario,  # pyright: ignore[reportAttributeAccessIssue]
)


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "dbos_test.sqlite3"


# =============================================================================
# Test 1: Basic interrupt/resume with input events
# =============================================================================


def test_determinism_on_resume_after_interrupt(test_db_path: Path) -> None:
    """Test that resuming an interrupted workflow doesn't hit determinism errors."""
    run_id = "test-determinism-001"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    result1 = run_scenario(
        workflow="tests.fixtures.sample_workflows.hitl:TestWorkflow",
        db_url=db_url,
        run_id=run_id,
        config={"interrupt_on": "AskInputEvent"},
    )
    log_on_failure(result1, "initial run")

    assert "STEP:ask:complete" in result1.stdout, "First step should complete"
    assert "INTERRUPTING" in result1.stdout, "Should have interrupted"

    result2 = run_scenario(
        workflow="tests.fixtures.sample_workflows.hitl:TestWorkflow",
        db_url=db_url,
        run_id=run_id,
        config={
            "respond": {
                "AskInputEvent": {
                    "event": "UserInput",
                    "fields": {"response": "test_input"},
                }
            }
        },
    )
    log_on_failure(result2, "resume")

    assert_no_determinism_errors(result2)
    assert "SUCCESS" in result2.stdout, (
        f"Resume should succeed. stdout: {result2.stdout}, stderr: {result2.stderr}"
    )


# =============================================================================
# Test 2: Chained steps determinism
# =============================================================================


def test_chained_steps_determinism_on_resume(test_db_path: Path) -> None:
    """Test determinism with chained steps that trigger each other."""
    run_id = "test-chained-001"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    result1 = run_scenario(
        workflow="tests.fixtures.sample_workflows.chained:ChainedWorkflow",
        db_url=db_url,
        run_id=run_id,
        config={"interrupt_on": "StepTwoEvent"},
    )
    log_on_failure(result1, "initial run")

    assert "STEP:one:complete" in result1.stdout, "Step one should complete"

    result2 = run_scenario(
        workflow="tests.fixtures.sample_workflows.chained:ChainedWorkflow",
        db_url=db_url,
        run_id=run_id,
    )
    log_on_failure(result2, "resume")

    assert_no_determinism_errors(result2)


# =============================================================================
# Test 3: Three-step HITL pattern
# =============================================================================


def test_hitl_three_step_determinism(test_db_path: Path) -> None:
    """Test the exact HITL pattern with three steps and input events."""
    run_id = "test-hitl-three-001"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    result1 = run_scenario(
        workflow="tests.fixtures.sample_workflows.three_step_hitl:HITLWorkflow",
        db_url=db_url,
        run_id=run_id,
        config={
            "respond": {
                "NameInputEvent": {
                    "event": "NameResponseEvent",
                    "fields": {"response": "Alice"},
                }
            },
            "interrupt_on": "QuestInputEvent",
        },
    )
    log_on_failure(result1, "initial run")

    assert "STEP:ask_name:complete" in result1.stdout, "ask_name should complete"
    assert "STEP:ask_quest" in result1.stdout, "ask_quest should start"
    assert "INTERRUPTING" in result1.stdout, "Should interrupt at quest"

    result2 = run_scenario(
        workflow="tests.fixtures.sample_workflows.three_step_hitl:HITLWorkflow",
        db_url=db_url,
        run_id=run_id,
        config={
            "respond": {
                "NameInputEvent": {
                    "event": "NameResponseEvent",
                    "fields": {"response": "Alice"},
                },
                "QuestInputEvent": {
                    "event": "QuestResponseEvent",
                    "fields": {"response": "seek the grail"},
                },
            },
        },
    )
    log_on_failure(result2, "resume")

    assert_no_determinism_errors(result2)
    assert "SUCCESS" in result2.stdout, (
        f"Resume should succeed.\nstdout: {result2.stdout}\nstderr: {result2.stderr}"
    )


# =============================================================================
# Test 4: Parallel steps - two steps triggered by StartEvent
# =============================================================================


def test_parallel_steps_determinism(test_db_path: Path) -> None:
    """Test determinism with parallel steps completing in non-deterministic order."""
    run_id = "test-parallel-001"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    result1 = run_scenario(
        workflow="tests.fixtures.sample_workflows.parallel:ParallelWorkflow",
        db_url=db_url,
        run_id=run_id,
    )
    log_on_failure(result1, "parallel run")

    assert "SUCCESS" in result1.stdout, (
        f"Should complete successfully.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
    )
    assert_no_determinism_errors(result1)


# =============================================================================
# Test 5: Concurrent workers on same step (num_workers=2)
# =============================================================================


def test_concurrent_workers_determinism(test_db_path: Path) -> None:
    """Test determinism with multiple workers on same step (num_workers > 1)."""
    run_id = "test-concurrent-workers-001"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    result1 = run_scenario(
        workflow="tests.fixtures.sample_workflows.concurrent_workers:ConcurrentWorkersWorkflow",
        db_url=db_url,
        run_id=run_id,
    )
    log_on_failure(result1, "concurrent workers run")

    assert "SUCCESS" in result1.stdout, (
        f"Should complete successfully.\nstdout: {result1.stdout}\nstderr: {result1.stderr}"
    )
    assert_no_determinism_errors(result1)


# =============================================================================
# Test 6: Sequential steps with HITL
# =============================================================================


def test_sequential_hitl_interrupt_resume(test_db_path: Path) -> None:
    """Test sequential steps with HITL interrupt and resume."""
    run_id = "test-seq-hitl-001"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    result1 = run_scenario(
        workflow="tests.fixtures.sample_workflows.sequential_hitl:SequentialHITLWorkflow",
        db_url=db_url,
        run_id=run_id,
        config={"interrupt_on": "WaitForInputEvent"},
    )
    log_on_failure(result1, "initial run")

    assert "STEP:process:complete" in result1.stdout
    assert "INTERRUPTING" in result1.stdout

    result2 = run_scenario(
        workflow="tests.fixtures.sample_workflows.sequential_hitl:SequentialHITLWorkflow",
        db_url=db_url,
        run_id=run_id,
        config={
            "respond": {
                "WaitForInputEvent": {
                    "event": "UserContinueEvent",
                    "fields": {"continue_value": "user_input"},
                }
            }
        },
    )
    log_on_failure(result2, "resume")

    assert_no_determinism_errors(result2)
    assert "SUCCESS" in result2.stdout, (
        f"Resume should succeed.\nstdout: {result2.stdout}\nstderr: {result2.stderr}"
    )


# =============================================================================
# Stress tests - run scenarios multiple times to catch flaky timing issues
# =============================================================================


@pytest.mark.timeout(60)
@pytest.mark.parametrize("iteration", range(5))
def test_parallel_steps_stress(test_db_path: Path, iteration: int) -> None:
    """Stress test parallel steps - run 5 times to catch timing issues."""
    run_id = f"test-parallel-stress-{iteration}"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    result = run_scenario(
        workflow="tests.fixtures.sample_workflows.parallel:ParallelWorkflow",
        db_url=db_url,
        run_id=run_id,
    )

    assert "SUCCESS" in result.stdout, (
        f"Iteration {iteration} failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert_no_determinism_errors(result)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("iteration", range(5))
def test_concurrent_workers_stress(test_db_path: Path, iteration: int) -> None:
    """Stress test concurrent workers - run 5 times to catch timing issues."""
    run_id = f"test-concurrent-stress-{iteration}"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    result = run_scenario(
        workflow="tests.fixtures.sample_workflows.concurrent_workers:ConcurrentWorkersWorkflow",
        db_url=db_url,
        run_id=run_id,
    )

    assert "SUCCESS" in result.stdout, (
        f"Iteration {iteration} failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert_no_determinism_errors(result)


# =============================================================================
# Test 7: Streaming stress test
# =============================================================================


def test_streaming_stress_determinism(test_db_path: Path) -> None:
    """Test determinism with many concurrent stream writes and send_event calls."""
    run_id = "test-streaming-stress-001"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    result = run_scenario(
        workflow="tests.fixtures.sample_workflows.streaming_stress:StreamingStressWorkflow",
        db_url=db_url,
        run_id=run_id,
    )
    log_on_failure(result, "streaming stress")

    assert "SUCCESS" in result.stdout, (
        f"Should complete successfully.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert_no_determinism_errors(result)


def test_streaming_interrupt_resume(test_db_path: Path) -> None:
    """Test interrupt/resume with many concurrent stream writes in flight."""
    run_id = "test-streaming-interrupt-001"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    result1 = run_scenario(
        workflow="tests.fixtures.sample_workflows.streaming_interrupt:StreamingInterruptWorkflow",
        db_url=db_url,
        run_id=run_id,
        config={
            "interrupt_on": {"event": "ProgressEvent", "condition": {"progress": 999}}
        },
    )
    log_on_failure(result1, "initial run")

    assert "STEP:fan_out:dispatched_15_items" in result1.stdout, (
        "Fan out should complete"
    )
    assert "INTERRUPTING" in result1.stdout, "Should have interrupted"

    result2 = run_scenario(
        workflow="tests.fixtures.sample_workflows.streaming_interrupt:StreamingInterruptWorkflow",
        db_url=db_url,
        run_id=run_id,
    )
    log_on_failure(result2, "resume")

    assert_no_determinism_errors(result2)
    assert "SUCCESS" in result2.stdout, (
        f"Resume should succeed.\nstdout: {result2.stdout}\nstderr: {result2.stderr}"
    )


@pytest.mark.timeout(60)
@pytest.mark.parametrize("iteration", range(5))
def test_streaming_stress_repeated(test_db_path: Path, iteration: int) -> None:
    """Stress test streaming - run 5 times to catch timing issues."""
    run_id = f"test-streaming-repeated-{iteration}"
    db_url = f"sqlite+pysqlite:///{test_db_path}?check_same_thread=false"

    result = run_scenario(
        workflow="tests.fixtures.sample_workflows.streaming_stress:StreamingStressWorkflow",
        db_url=db_url,
        run_id=run_id,
    )

    assert "SUCCESS" in result.stdout, (
        f"Iteration {iteration} failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert_no_determinism_errors(result)
