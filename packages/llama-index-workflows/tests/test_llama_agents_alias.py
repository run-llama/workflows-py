# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests that the llama_agents.workflows alias resolves correctly."""

from __future__ import annotations

import pytest


def test_top_level_import() -> None:
    from llama_agents.workflows import Context, Workflow, step

    assert Workflow is not None
    assert Context is not None
    assert callable(step)


def test_top_level_identity() -> None:
    import llama_agents.workflows as alias
    import workflows

    assert alias.Workflow is workflows.Workflow
    assert alias.Context is workflows.Context
    assert alias.step is workflows.step


def test_events_submodule() -> None:
    from llama_agents.workflows.events import (
        Event,
        StartEvent,
        StopEvent,
    )

    assert Event is not None
    assert issubclass(StartEvent, Event)
    assert issubclass(StopEvent, Event)


def test_events_submodule_identity() -> None:
    import llama_agents.workflows.events as alias_events
    import workflows.events

    assert alias_events is workflows.events
    assert alias_events.Event is workflows.events.Event


def test_context_submodule() -> None:
    from llama_agents.workflows.context import Context

    assert Context is not None


def test_context_submodule_identity() -> None:
    import llama_agents.workflows.context as alias_ctx
    import workflows.context

    assert alias_ctx is workflows.context


def test_workflow_submodule() -> None:
    from llama_agents.workflows.workflow import Workflow

    assert Workflow is not None


def test_decorators_submodule() -> None:
    from llama_agents.workflows.decorators import step

    assert callable(step)


def test_errors_submodule() -> None:
    from llama_agents.workflows.errors import (
        WorkflowRuntimeError,
        WorkflowTimeoutError,
        WorkflowValidationError,
    )

    assert issubclass(WorkflowRuntimeError, Exception)
    assert issubclass(WorkflowTimeoutError, Exception)
    assert issubclass(WorkflowValidationError, Exception)


def test_handler_submodule() -> None:
    from llama_agents.workflows.handler import WorkflowHandler

    assert WorkflowHandler is not None


def test_testing_submodule() -> None:
    from llama_agents.workflows.testing import WorkflowTestRunner

    assert WorkflowTestRunner is not None


def test_deep_submodule() -> None:
    from llama_agents.workflows.context.context import Context

    assert Context is not None


def test_dunder_all_reexported() -> None:
    import llama_agents.workflows as alias
    import workflows

    assert alias.__all__ == workflows.__all__


@pytest.mark.asyncio
async def test_alias_workflow_runs() -> None:
    """A workflow defined via the alias module actually executes."""
    from llama_agents.workflows import Context, Workflow, step
    from llama_agents.workflows.events import StartEvent, StopEvent

    class HelloWorkflow(Workflow):
        @step
        async def say_hello(self, ctx: Context, ev: StartEvent) -> StopEvent:
            return StopEvent(result="hello")

    wf = HelloWorkflow()
    result = await wf.run()
    assert result == "hello"
