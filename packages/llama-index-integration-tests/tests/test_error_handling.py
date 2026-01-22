"""Tests for workflow error handling.

These tests verify that error conditions like max iterations are
handled correctly, and that early stopping methods work as expected.
"""

import pytest
from llama_index_integration_tests.helpers import (
    make_text_response,
    make_tool_call_response,
)
from workflows.errors import WorkflowRuntimeError

from .conftest import WorkflowFactory


async def test_max_iterations_raises_error(create_workflow: WorkflowFactory) -> None:
    """Test that exceeding max_iterations raises WorkflowRuntimeError."""

    def infinite_tool() -> str:
        """A tool that will be called repeatedly."""
        return "called"

    workflow = create_workflow(
        tools=[infinite_tool],
        # Response generator that always calls the tool
        responses=[make_tool_call_response("infinite_tool")] * 100,
    )

    with pytest.raises(WorkflowRuntimeError, match="Max iterations"):
        await workflow.run(user_msg="Loop forever", max_iterations=5)


async def test_early_stopping_generate(create_workflow: WorkflowFactory) -> None:
    """Test early_stopping_method='generate' produces a final response instead of error."""

    def looping_tool() -> str:
        return "still going"

    workflow = create_workflow(
        tools=[looping_tool],
        responses=[
            make_tool_call_response("looping_tool"),
            make_tool_call_response("looping_tool"),
            make_tool_call_response("looping_tool"),
            make_tool_call_response("looping_tool"),
            make_tool_call_response("looping_tool"),
            # After max iterations, LLM should give final response
            make_text_response("Final summary after hitting limit"),
        ],
        early_stopping_method="generate",
    )

    # Should NOT raise, should return a response
    handler = workflow.run(user_msg="Test", max_iterations=5)
    async for _ in handler.stream_events():
        pass

    result = await handler
    assert result is not None


async def test_tool_error_captured_in_result(create_workflow: WorkflowFactory) -> None:
    """Test that tool errors are captured rather than crashing the workflow."""

    def failing_tool() -> str:
        raise ValueError("Tool failed!")

    workflow = create_workflow(
        tools=[failing_tool],
        responses=[
            make_tool_call_response("failing_tool"),
            make_text_response("Handled the error"),
        ],
    )

    handler = workflow.run(user_msg="Use the failing tool")
    async for _ in handler.stream_events():
        pass

    # Workflow should complete (error is passed back to LLM)
    result = await handler
    assert result is not None


async def test_max_iterations_configurable_per_run(
    create_workflow: WorkflowFactory,
) -> None:
    """Test that max_iterations can be set per run() call."""
    call_count = 0

    def counting_tool() -> str:
        nonlocal call_count
        call_count += 1
        return f"Call {call_count}"

    workflow = create_workflow(
        tools=[counting_tool],
        responses=[make_tool_call_response("counting_tool")] * 50
        + [make_text_response("Done")],
    )

    # With low max_iterations, should fail
    with pytest.raises(WorkflowRuntimeError):
        await workflow.run(user_msg="Test", max_iterations=3)

    # Reset
    call_count = 0

    # With higher max_iterations, more calls happen before error
    with pytest.raises(WorkflowRuntimeError):
        await workflow.run(user_msg="Test", max_iterations=10)

    # Should have made more calls with higher limit
    assert call_count > 3
