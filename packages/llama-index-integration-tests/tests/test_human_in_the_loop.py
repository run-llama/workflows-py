"""Tests for human-in-the-loop (HITL) workflow patterns.

These tests verify that wait_for_event works correctly inside tool
functions, and that workflows can be paused, serialized, and resumed.
"""

from conftest import WorkflowFactory
from llama_index_integration_tests.helpers import (
    make_text_response,
    make_tool_call_response,
)
from workflows import Context
from workflows.context import PickleSerializer
from workflows.events import HumanResponseEvent, InputRequiredEvent


async def test_wait_for_event_in_tool(create_workflow: WorkflowFactory) -> None:
    """Test that wait_for_event works inside a tool function."""
    received_response = None

    async def ask_human(ctx: Context) -> str:
        nonlocal received_response
        resp = await ctx.wait_for_event(
            HumanResponseEvent,
            waiter_event=InputRequiredEvent(prefix="What is your name?"),  # type: ignore[call-arg]
        )
        received_response = resp.response
        return f"Hello, {resp.response}!"

    workflow = create_workflow(
        tools=[ask_human],
        responses=[
            make_tool_call_response("ask_human"),
            make_text_response("Greeted the user"),
        ],
    )

    handler = workflow.run(user_msg="Ask for my name")

    # Wait for the InputRequiredEvent
    paused = False
    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            assert event.prefix == "What is your name?"
            paused = True
            await handler.cancel_run()
            break

    assert paused, "Should have paused for human input"

    # Resume with human response
    new_ctx = Context.from_dict(workflow, handler.ctx.to_dict())
    handler = workflow.run(user_msg="Ask for my name", ctx=new_ctx)
    handler.ctx.send_event(HumanResponseEvent(response="Alice"))  # type: ignore[call-arg]

    await handler

    assert received_response == "Alice"


async def test_context_dict_serialization(create_workflow: WorkflowFactory) -> None:
    """Test that context can be serialized to dict and restored."""

    async def pausable_tool(ctx: Context) -> str:
        await ctx.wait_for_event(
            HumanResponseEvent,
            waiter_event=InputRequiredEvent(prefix="Continue?"),  # type: ignore[call-arg]
        )
        return "Continued"

    workflow = create_workflow(
        tools=[pausable_tool],
        responses=[
            make_tool_call_response("pausable_tool"),
            make_text_response("Done"),
        ],
    )

    handler = workflow.run(user_msg="Test")

    # Pause and serialize
    ctx_dict = None
    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            ctx_dict = handler.ctx.to_dict()
            await handler.cancel_run()
            break

    assert ctx_dict is not None

    # Restore and resume
    restored_ctx = Context.from_dict(workflow, ctx_dict)
    handler = workflow.run(user_msg="Test", ctx=restored_ctx)
    handler.ctx.send_event(HumanResponseEvent(response="yes"))  # type: ignore[call-arg]

    result = await handler
    assert result is not None


async def test_pickle_serialization(create_workflow: WorkflowFactory) -> None:
    """Test that context can be pickle-serialized for cross-process persistence."""

    async def pausable_tool(ctx: Context) -> str:
        await ctx.wait_for_event(
            HumanResponseEvent,
            waiter_event=InputRequiredEvent(prefix="Confirm?"),  # type: ignore[call-arg]
        )
        return "Confirmed"

    workflow = create_workflow(
        tools=[pausable_tool],
        responses=[
            make_tool_call_response("pausable_tool"),
            make_text_response("Done"),
        ],
    )

    handler = workflow.run(user_msg="Test")

    # Pause and pickle-serialize
    ctx_dict = None
    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            serializer = PickleSerializer()
            ctx_dict = handler.ctx.to_dict(serializer=serializer)
            await handler.cancel_run()
            break

    assert ctx_dict is not None

    # Restore with pickle deserializer
    serializer = PickleSerializer()
    restored_ctx = Context.from_dict(workflow, ctx_dict, serializer=serializer)
    handler = workflow.run(user_msg="Test", ctx=restored_ctx)
    handler.ctx.send_event(HumanResponseEvent(response="confirmed"))  # type: ignore[call-arg]

    result = await handler
    assert result is not None


async def test_state_preserved_across_pause_resume(
    create_workflow: WorkflowFactory,
) -> None:
    """Test that workflow state is preserved when pausing and resuming."""

    async def stateful_pause(ctx: Context) -> str:
        # Modify state before pausing
        state = await ctx.store.get("state")
        state["before_pause"] = True
        await ctx.store.set("state", state)

        await ctx.wait_for_event(
            HumanResponseEvent,
            waiter_event=InputRequiredEvent(prefix="Ready?"),  # type: ignore[call-arg]
        )

        # State should still be there after resume
        state = await ctx.store.get("state")
        state["after_resume"] = True
        await ctx.store.set("state", state)

        return "Done"

    workflow = create_workflow(
        tools=[stateful_pause],
        responses=[
            make_tool_call_response("stateful_pause"),
            make_text_response("Complete"),
        ],
        initial_state={"initial": True},
    )

    handler = workflow.run(user_msg="Test")

    # Pause
    ctx_dict = None
    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            ctx_dict = handler.ctx.to_dict()
            await handler.cancel_run()
            break

    assert ctx_dict is not None

    # Resume
    restored_ctx = Context.from_dict(workflow, ctx_dict)
    handler = workflow.run(user_msg="Test", ctx=restored_ctx)
    handler.ctx.send_event(HumanResponseEvent(response="yes"))  # type: ignore[call-arg]

    await handler

    # Check all state was preserved
    state = await handler.ctx.store.get("state")
    assert state["initial"] is True
    assert state["before_pause"] is True
    assert state["after_resume"] is True
