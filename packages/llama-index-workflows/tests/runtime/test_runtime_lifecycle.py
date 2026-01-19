# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for Runtime lifecycle (registering context manager, context variable)."""

from __future__ import annotations

from typing import cast

import pytest
from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent
from workflows.plugins import BasicRuntime, basic_runtime, get_current_runtime
from workflows.runtime.types.plugin import (
    ControlLoopFunction,
    Runtime,
    _current_runtime,
)


class SimpleWorkflow(Workflow):
    @step
    async def start(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="done")


def test_basic_runtime_is_runtime_instance() -> None:
    """BasicRuntime extends Runtime ABC."""
    assert isinstance(basic_runtime, Runtime)


def test_basic_runtime_has_lifecycle_methods() -> None:
    """BasicRuntime has launch() and destroy() methods."""
    runtime = BasicRuntime()
    # Should not raise
    runtime.launch()
    runtime.destroy()


def test_get_current_runtime_returns_basic_runtime_by_default() -> None:
    """When no context-scoped runtime, get_current_runtime returns basic_runtime."""
    runtime = get_current_runtime()
    assert runtime is basic_runtime


def test_registering_sets_current_runtime() -> None:
    """registering() context manager sets the context-scoped runtime."""
    custom_runtime = BasicRuntime()

    with custom_runtime.registering():
        current = _current_runtime.get()
        assert current is custom_runtime


def test_registering_resets_on_exit() -> None:
    """registering() context manager resets the context variable on exit."""
    custom_runtime = BasicRuntime()

    with custom_runtime.registering():
        pass

    current = _current_runtime.get()
    assert current is None


def test_registering_returns_runtime() -> None:
    """registering() context manager returns the runtime when entering."""
    custom_runtime = BasicRuntime()

    with custom_runtime.registering() as r:
        assert r is custom_runtime


def test_registering_does_not_call_launch_on_exit() -> None:
    """registering() context manager does NOT call launch() on exit."""
    launched = False

    class TrackingRuntime(BasicRuntime):
        def launch(self) -> None:
            nonlocal launched
            launched = True

    with TrackingRuntime().registering():
        pass

    # Key change: launch() should NOT be called
    assert not launched


def test_registering_does_not_call_launch_on_exception() -> None:
    """registering() context manager does NOT call launch() when exception is raised."""
    launched = False

    class TrackingRuntime(BasicRuntime):
        def launch(self) -> None:
            nonlocal launched
            launched = True

    with pytest.raises(ValueError):
        with TrackingRuntime().registering():
            raise ValueError("test error")

    assert not launched


def test_get_current_runtime_returns_context_runtime_when_set() -> None:
    """get_current_runtime returns context-scoped runtime when set."""
    custom_runtime = BasicRuntime()

    with custom_runtime.registering():
        assert get_current_runtime() is custom_runtime


def test_nested_registering_contexts() -> None:
    """Nested registering() context managers restore correctly."""
    outer_runtime = BasicRuntime()
    inner_runtime = BasicRuntime()

    with outer_runtime.registering():
        assert get_current_runtime() is outer_runtime

        with inner_runtime.registering():
            assert get_current_runtime() is inner_runtime

        # After inner exits, should restore to outer
        assert get_current_runtime() is outer_runtime

    # After outer exits, should restore to basic_runtime
    assert get_current_runtime() is basic_runtime


def test_basic_runtime_register_returns_none() -> None:
    """BasicRuntime.register() returns None (no wrapping needed)."""
    wf = SimpleWorkflow()
    runtime = BasicRuntime()

    # Mock control loop - cast to protocol since we only test the return value
    async def mock_control_loop(
        start_event: object, init_state: object, run_id: str
    ) -> StopEvent:
        return StopEvent(result="done")

    result = runtime.register(wf, cast(ControlLoopFunction, mock_control_loop), {})
    assert result is None


def test_explicit_runtime_parameter() -> None:
    """Workflow with runtime= uses that runtime."""
    custom_runtime = BasicRuntime()
    wf = SimpleWorkflow(runtime=custom_runtime)
    assert wf.runtime is custom_runtime


def test_registering_context_manager() -> None:
    """Workflows inside registering() use context-scoped runtime."""
    custom_runtime = BasicRuntime()

    with custom_runtime.registering():
        wf = SimpleWorkflow()
        assert wf.runtime is custom_runtime


def test_explicit_overrides_registering() -> None:
    """Explicit runtime= takes precedence over registering() context."""
    context_runtime = BasicRuntime()
    explicit_runtime = BasicRuntime()

    with context_runtime.registering():
        wf = SimpleWorkflow(runtime=explicit_runtime)
        assert wf.runtime is explicit_runtime


def test_fallback_to_basic_runtime() -> None:
    """Workflow without runtime uses basic_runtime."""
    wf = SimpleWorkflow()
    assert wf.runtime is basic_runtime


def test_workflow_runs_after_registering_exit() -> None:
    """Workflow can run after registering() exits (requires explicit launch())."""
    custom_runtime = BasicRuntime()

    with custom_runtime.registering():
        SimpleWorkflow()

    # Explicit launch() should be called before running
    custom_runtime.launch()
    # BasicRuntime doesn't actually need launch(), but the pattern should work


def test_registering_yields_runtime() -> None:
    """registering() context manager yields the runtime."""
    custom_runtime = BasicRuntime()

    with custom_runtime.registering() as r:
        assert r is custom_runtime
        assert isinstance(r, Runtime)


def test_empty_registering_block() -> None:
    """Empty registering() block is valid no-op."""
    custom_runtime = BasicRuntime()

    # Should not raise
    with custom_runtime.registering():
        pass

    # Context should be reset
    assert get_current_runtime() is basic_runtime


def test_registering_with_exception_still_resets_context() -> None:
    """Context reset even on exception inside registering()."""
    custom_runtime = BasicRuntime()

    with pytest.raises(RuntimeError):
        with custom_runtime.registering():
            assert get_current_runtime() is custom_runtime
            raise RuntimeError("test")

    # Context should still be reset after exception
    assert get_current_runtime() is basic_runtime
