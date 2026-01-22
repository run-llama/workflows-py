"""Tests for workflow event streaming.

These tests verify that event streaming works correctly with llama-index
agents, including write_event_to_stream, collect_events, and the
stream_events() API.
"""

from conftest import SimpleWorkflowFactory, WorkflowFactory
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult
from llama_index_integration_tests.helpers import (
    make_text_response,
    make_tool_call_response,
)


async def test_stream_events_yields_events(
    create_simple_workflow: SimpleWorkflowFactory,
) -> None:
    """Test that stream_events() yields events during workflow execution."""
    workflow = create_simple_workflow(
        responses=[make_text_response("Hello!")],
    )

    handler = workflow.run(user_msg="Hi")

    events = []
    async for event in handler.stream_events():
        events.append(event)

    await handler

    # Should have received some events
    assert len(events) > 0


async def test_agent_output_streamed(
    create_simple_workflow: SimpleWorkflowFactory,
) -> None:
    """Test that AgentOutput events are streamed."""
    workflow = create_simple_workflow(
        responses=[make_text_response("Test response")],
    )

    handler = workflow.run(user_msg="Test")

    agent_outputs = []
    async for event in handler.stream_events():
        if isinstance(event, AgentOutput):
            agent_outputs.append(event)

    await handler

    # Should have at least one AgentOutput
    assert len(agent_outputs) >= 1


async def test_tool_call_events_streamed(create_workflow: WorkflowFactory) -> None:
    """Test that ToolCall and ToolCallResult events are streamed."""

    def dummy_tool() -> str:
        """A dummy tool."""
        return "tool result"

    workflow = create_workflow(
        tools=[dummy_tool],
        responses=[
            make_tool_call_response("dummy_tool"),
            make_text_response("Done"),
        ],
    )

    handler = workflow.run(user_msg="Use the tool")

    tool_calls = []
    tool_results = []
    async for event in handler.stream_events():
        if isinstance(event, ToolCall):
            tool_calls.append(event)
        elif isinstance(event, ToolCallResult):
            tool_results.append(event)

    await handler

    # Should have tool call and result events
    assert len(tool_calls) >= 1
    assert len(tool_results) >= 1


async def test_multiple_tool_calls_streamed(create_workflow: WorkflowFactory) -> None:
    """Test that multiple sequential tool calls each produce events."""
    call_count = 0

    def counting_tool() -> str:
        nonlocal call_count
        call_count += 1
        return f"Call {call_count}"

    workflow = create_workflow(
        tools=[counting_tool],
        responses=[
            make_tool_call_response("counting_tool", tool_id="call_1"),
            make_tool_call_response("counting_tool", tool_id="call_2"),
            make_text_response("Done"),
        ],
    )

    handler = workflow.run(user_msg="Call tool twice")

    tool_results = []
    async for event in handler.stream_events():
        if isinstance(event, ToolCallResult):
            tool_results.append(event)

    await handler

    # Should have two tool results
    assert len(tool_results) == 2
    assert call_count == 2


async def test_handler_result_after_streaming(
    create_simple_workflow: SimpleWorkflowFactory,
) -> None:
    """Test that awaiting handler after streaming returns the final result."""
    workflow = create_simple_workflow(
        responses=[make_text_response("Final answer")],
    )

    handler = workflow.run(user_msg="Question")

    # Stream all events
    async for _ in handler.stream_events():
        pass

    # Await should return the result
    result = await handler

    assert result is not None
    assert isinstance(result, AgentOutput)


async def test_events_contain_agent_name(
    create_simple_workflow: SimpleWorkflowFactory,
) -> None:
    """Test that streamed events contain the current agent name."""
    workflow = create_simple_workflow(
        name="named_agent",
        responses=[make_text_response("Response")],
    )

    handler = workflow.run(user_msg="Test")

    events_with_name = []
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name"):
            events_with_name.append(event)

    await handler

    # Events should have the agent name
    assert len(events_with_name) > 0
    for event in events_with_name:
        assert event.current_agent_name == "named_agent"
