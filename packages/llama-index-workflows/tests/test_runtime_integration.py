# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Integration tests for runtime lifecycle with registering() context manager."""

from __future__ import annotations

from typing import Any

import pytest
from workflows import Context, Workflow, step
from workflows.events import StartEvent, StopEvent
from workflows.plugins import BasicRuntime, basic_runtime, get_current_runtime
from workflows.testing import WorkflowTestRunner


class SimpleWorkflow(Workflow):
    """Simple test workflow."""

    @step
    async def start(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="done")


class CountingWorkflow(Workflow):
    """Workflow that counts invocations."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run_count = 0

    @step
    async def start(self, ev: StartEvent) -> StopEvent:
        self.run_count += 1
        return StopEvent(result=self.run_count)


class StatefulWorkflow(Workflow):
    """Workflow that preserves state across runs."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value = 0

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> StopEvent:
        if hasattr(ev, "value"):
            self.value = ev.value  # type: ignore[attr-defined]
        return StopEvent(result=self.value)


# Basic Runtime Tests


async def test_basic_runtime_no_explicit_launch() -> None:
    """BasicRuntime works without explicit launch()."""
    wf = SimpleWorkflow()
    result = await WorkflowTestRunner(wf).run()
    assert result.result == "done"


async def test_basic_runtime_with_explicit_launch() -> None:
    """launch()/destroy() are no-ops but don't error for BasicRuntime."""
    runtime = BasicRuntime()
    runtime.launch()

    wf = SimpleWorkflow(runtime=runtime)
    result = await WorkflowTestRunner(wf).run()
    assert result.result == "done"

    runtime.destroy()


async def test_basic_runtime_launch_idempotent() -> None:
    """Multiple launch() calls are allowed."""
    runtime = BasicRuntime()
    runtime.launch()
    runtime.launch()

    wf = SimpleWorkflow(runtime=runtime)
    result = await WorkflowTestRunner(wf).run()
    assert result.result == "done"


# Registering Context Manager Tests


async def test_registering_multiple_workflows() -> None:
    """Multiple workflows register in same block."""
    runtime = BasicRuntime()

    with runtime.registering():
        wf1 = SimpleWorkflow()
        wf2 = CountingWorkflow()

    assert wf1.runtime is runtime
    assert wf2.runtime is runtime

    runtime.launch()

    result1 = await WorkflowTestRunner(wf1).run()
    result2 = await WorkflowTestRunner(wf2).run()

    assert result1.result == "done"
    assert result2.result == 1


async def test_registering_preserves_workflow_state() -> None:
    """Workflow state preserved through launch()."""
    runtime = BasicRuntime()

    with runtime.registering():
        wf = CountingWorkflow()

    runtime.launch()

    # Run multiple times, state should persist
    result1 = await WorkflowTestRunner(wf).run()
    result2 = await WorkflowTestRunner(wf).run()

    assert result1.result == 1
    assert result2.result == 2


async def test_registering_with_exception_still_resets_context() -> None:
    """Context reset even on exception."""
    runtime = BasicRuntime()

    with pytest.raises(ValueError):
        with runtime.registering():
            raise ValueError("test")

    # Context should be reset
    assert get_current_runtime() is basic_runtime


async def test_mixed_explicit_and_implicit_registration() -> None:
    """Explicit runtime= overrides context."""
    context_runtime = BasicRuntime()
    explicit_runtime = BasicRuntime()

    with context_runtime.registering():
        wf_implicit = SimpleWorkflow()
        wf_explicit = SimpleWorkflow(runtime=explicit_runtime)

    assert wf_implicit.runtime is context_runtime
    assert wf_explicit.runtime is explicit_runtime

    context_runtime.launch()
    explicit_runtime.launch()

    result1 = await WorkflowTestRunner(wf_implicit).run()
    result2 = await WorkflowTestRunner(wf_explicit).run()

    assert result1.result == "done"
    assert result2.result == "done"


# Workflow Execution Tests


async def test_workflow_can_run_multiple_times() -> None:
    """Same workflow runs multiple times after launch()."""
    runtime = BasicRuntime()

    with runtime.registering():
        wf = CountingWorkflow()

    runtime.launch()

    result1 = await WorkflowTestRunner(wf).run()
    result2 = await WorkflowTestRunner(wf).run()
    result3 = await WorkflowTestRunner(wf).run()

    assert result1.result == 1
    assert result2.result == 2
    assert result3.result == 3


async def test_destroy_allows_reuse() -> None:
    """Runtime can create new workflows after destroy()."""
    runtime = BasicRuntime()

    with runtime.registering():
        wf1 = SimpleWorkflow()

    runtime.launch()
    result1 = await WorkflowTestRunner(wf1).run()

    runtime.destroy()

    # Can register new workflows after destroy
    with runtime.registering():
        wf2 = SimpleWorkflow()

    runtime.launch()
    result2 = await WorkflowTestRunner(wf2).run()

    assert result1.result == "done"
    assert result2.result == "done"


# Edge Cases


async def test_workflow_without_any_runtime_context() -> None:
    """Falls back to basic_runtime."""
    wf = SimpleWorkflow()
    assert wf.runtime is basic_runtime

    result = await WorkflowTestRunner(wf).run()
    assert result.result == "done"


async def test_empty_registering_block() -> None:
    """Empty block is valid no-op."""
    runtime = BasicRuntime()

    with runtime.registering():
        pass

    # Context should be reset
    assert get_current_runtime() is basic_runtime


async def test_registering_yields_runtime() -> None:
    """Context manager yields the runtime."""
    runtime = BasicRuntime()

    with runtime.registering() as r:
        assert r is runtime

    # Context should be reset after exit
    assert get_current_runtime() is basic_runtime


async def test_nested_registering_preserves_workflows() -> None:
    """Nested registering blocks correctly assign workflows."""
    outer_runtime = BasicRuntime()
    inner_runtime = BasicRuntime()

    with outer_runtime.registering():
        wf_outer = SimpleWorkflow()

        with inner_runtime.registering():
            wf_inner = SimpleWorkflow()

        wf_outer_again = SimpleWorkflow()

    assert wf_outer.runtime is outer_runtime
    assert wf_inner.runtime is inner_runtime
    assert wf_outer_again.runtime is outer_runtime

    outer_runtime.launch()
    inner_runtime.launch()

    r1 = await WorkflowTestRunner(wf_outer).run()
    r2 = await WorkflowTestRunner(wf_inner).run()
    r3 = await WorkflowTestRunner(wf_outer_again).run()

    assert r1.result == "done"
    assert r2.result == "done"
    assert r3.result == "done"


async def test_workflow_runs_without_registering() -> None:
    """Workflows created outside registering() use basic_runtime."""
    wf = SimpleWorkflow()
    assert wf.runtime is basic_runtime

    result = await WorkflowTestRunner(wf).run()
    assert result.result == "done"


async def test_multiple_concurrent_workflows() -> None:
    """Multiple workflows can run concurrently."""
    import asyncio

    runtime = BasicRuntime()

    with runtime.registering():
        wf1 = SimpleWorkflow()
        wf2 = SimpleWorkflow()
        wf3 = SimpleWorkflow()

    runtime.launch()

    # Run all concurrently
    results = await asyncio.gather(
        WorkflowTestRunner(wf1).run(),
        WorkflowTestRunner(wf2).run(),
        WorkflowTestRunner(wf3).run(),
    )

    assert results[0].result == "done"
    assert results[1].result == "done"
    assert results[2].result == "done"
