"""Tests for Plugin lifecycle (Phase 2: ABC, context manager, context variable)."""

from __future__ import annotations

import pytest
from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent
from workflows.plugins import BasicRuntime, basic_runtime, get_current_plugin
from workflows.runtime.types.plugin import Plugin, _current_plugin


class SimpleWorkflow(Workflow):
    @step
    async def start(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="done")


def test_basic_runtime_is_plugin_instance() -> None:
    """BasicRuntime extends Plugin ABC."""
    assert isinstance(basic_runtime, Plugin)


def test_basic_runtime_has_lifecycle_methods() -> None:
    """BasicRuntime has launch() and destroy() methods."""
    plugin = BasicRuntime()
    # Should not raise
    plugin.launch()
    plugin.destroy()


def test_get_current_plugin_returns_basic_runtime_by_default() -> None:
    """When no context-scoped plugin, get_current_plugin returns basic_runtime."""
    plugin = get_current_plugin()
    assert plugin is basic_runtime


def test_context_manager_sets_current_plugin() -> None:
    """Context manager sets the context-scoped plugin."""
    custom_plugin = BasicRuntime()

    with custom_plugin:
        current = _current_plugin.get()
        assert current is custom_plugin


def test_context_manager_resets_on_exit() -> None:
    """Context manager resets the context variable on exit."""
    custom_plugin = BasicRuntime()

    with custom_plugin:
        pass

    current = _current_plugin.get()
    assert current is None


def test_context_manager_returns_plugin() -> None:
    """Context manager returns the plugin when entering."""
    custom_plugin = BasicRuntime()

    with custom_plugin as p:
        assert p is custom_plugin


def test_context_manager_calls_launch_on_clean_exit() -> None:
    """Context manager calls launch() on clean exit."""
    launched = False

    class TrackingPlugin(BasicRuntime):
        def launch(self) -> None:
            nonlocal launched
            launched = True

    with TrackingPlugin():
        assert not launched

    assert launched


def test_context_manager_does_not_call_launch_on_exception() -> None:
    """Context manager does not call launch() when exception is raised."""
    launched = False

    class TrackingPlugin(BasicRuntime):
        def launch(self) -> None:
            nonlocal launched
            launched = True

    with pytest.raises(ValueError):
        with TrackingPlugin():
            raise ValueError("test error")

    assert not launched


def test_get_current_plugin_returns_context_plugin_when_set() -> None:
    """get_current_plugin returns context-scoped plugin when set."""
    custom_plugin = BasicRuntime()

    with custom_plugin:
        assert get_current_plugin() is custom_plugin


def test_nested_context_managers() -> None:
    """Nested context managers restore correctly."""
    outer_plugin = BasicRuntime()
    inner_plugin = BasicRuntime()

    with outer_plugin:
        assert get_current_plugin() is outer_plugin

        with inner_plugin:
            assert get_current_plugin() is inner_plugin

        # After inner exits, should restore to outer
        assert get_current_plugin() is outer_plugin

    # After outer exits, should restore to basic_runtime
    assert get_current_plugin() is basic_runtime


def test_basic_runtime_register_returns_none() -> None:
    """BasicRuntime.register() returns None (no wrapping needed)."""
    wf = SimpleWorkflow()
    plugin = BasicRuntime()

    # Mock control loop and steps
    async def control_loop(
        start_event: object, init_state: object, run_id: str
    ) -> StopEvent:
        return StopEvent(result="done")

    result = plugin.register(wf, control_loop, {})  # type: ignore[arg-type]
    assert result is None
