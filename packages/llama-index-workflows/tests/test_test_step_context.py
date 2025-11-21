# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field
from workflows import Context, Workflow, step
from workflows.context.state_store import DictState
from workflows.events import Event, StartEvent, StopEvent
from workflows.testing import TestStepContext


class CustomEvent(Event):
    count: int


class ProgressEvent(Event):
    message: str


class MyState(BaseModel):
    counter: int = Field(default=0)
    name: str = Field(default="test")


@pytest.mark.asyncio
async def test_create_with_no_state() -> None:
    """Test creating a TestStepContext without initial state."""
    ctx = TestStepContext.create()
    assert ctx is not None
    assert ctx.is_running is True
    assert len(ctx.sent_events) == 0
    assert len(ctx.streamed_events) == 0
    # Should have empty DictState by default
    state = await ctx.store.get_state()
    assert isinstance(state, DictState)


@pytest.mark.asyncio
async def test_create_with_dict_state() -> None:
    """Test creating a TestStepContext with DictState."""
    initial = DictState(count=10, message="hello")
    ctx = TestStepContext.create(initial)

    # Check initial state is set correctly
    count = await ctx.store.get("count")
    assert count == 10
    message = await ctx.store.get("message")
    assert message == "hello"


@pytest.mark.asyncio
async def test_create_with_typed_state() -> None:
    """Test creating a TestStepContext with a typed Pydantic model."""
    initial = MyState(counter=5, name="workflow")
    ctx = TestStepContext.create(initial)  # type: ignore[arg-type]

    # Check initial state is set correctly
    state = await ctx.store.get_state()
    assert isinstance(state, MyState)
    assert state.counter == 5
    assert state.name == "workflow"


@pytest.mark.asyncio
async def test_send_event_tracking() -> None:
    """Test that send_event properly tracks sent events."""
    ctx = TestStepContext.create()

    event1 = CustomEvent(count=1)
    event2 = CustomEvent(count=2)

    # Send events
    ctx.send_event(event1)
    ctx.send_event(event2, step="my_step")

    # Verify tracking
    assert len(ctx.sent_events) == 2
    assert ctx.sent_events[0] == (event1, None)
    assert ctx.sent_events[1] == (event2, "my_step")


@pytest.mark.asyncio
async def test_write_event_to_stream_tracking() -> None:
    """Test that write_event_to_stream properly tracks streamed events."""
    ctx = TestStepContext.create()

    event1 = ProgressEvent(message="starting")
    event2 = ProgressEvent(message="processing")

    # Write events to stream
    ctx.write_event_to_stream(event1)
    ctx.write_event_to_stream(event2)

    # Verify tracking
    assert len(ctx.streamed_events) == 2
    assert ctx.streamed_events[0] == event1
    assert ctx.streamed_events[1] == event2


@pytest.mark.asyncio
async def test_write_event_to_stream_ignores_none() -> None:
    """Test that write_event_to_stream ignores None values."""
    ctx = TestStepContext.create()

    event = ProgressEvent(message="test")

    # Write None and a real event
    ctx.write_event_to_stream(None)
    ctx.write_event_to_stream(event)
    ctx.write_event_to_stream(None)

    # Verify only the real event was tracked
    assert len(ctx.streamed_events) == 1
    assert ctx.streamed_events[0] == event


@pytest.mark.asyncio
async def test_clear_tracked_events() -> None:
    """Test that clear_tracked_events clears both event lists."""
    ctx = TestStepContext.create()

    # Add some events
    ctx.send_event(CustomEvent(count=1))
    ctx.write_event_to_stream(ProgressEvent(message="test"))

    assert len(ctx.sent_events) == 1
    assert len(ctx.streamed_events) == 1

    # Clear
    ctx.clear_tracked_events()

    assert len(ctx.sent_events) == 0
    assert len(ctx.streamed_events) == 0


@pytest.mark.asyncio
async def test_state_mutations() -> None:
    """Test that state can be mutated as expected."""
    ctx = TestStepContext.create(MyState(counter=0))  # type: ignore[arg-type]

    # Get and modify state
    state = await ctx.store.get_state()
    assert state.counter == 0

    # Set new state
    await ctx.store.set_state(MyState(counter=5, name="updated"))  # type: ignore[arg-type]

    # Verify change
    new_state = await ctx.store.get_state()
    assert new_state.counter == 5
    assert new_state.name == "updated"


@pytest.mark.asyncio
async def test_with_workflow_constructor() -> None:
    """Test creating TestStepContext with a workflow instance."""

    class TestWorkflow(Workflow):
        @step
        async def my_step(self, ctx: Context[MyState], ev: StartEvent) -> StopEvent:
            state = await ctx.store.get_state()
            return StopEvent(result=state.counter)

    workflow = TestWorkflow()
    initial = MyState(counter=42)
    ctx = TestStepContext(workflow, initial_state=initial)

    # Verify state was set
    state = await ctx.store.get_state()
    assert state.counter == 42


@pytest.mark.asyncio
async def test_step_function_with_test_context() -> None:
    """Test using TestStepContext to test a real step function."""

    class TestWorkflow(Workflow):
        @step
        async def my_step(self, ctx: Context[MyState], ev: StartEvent) -> CustomEvent:
            # Get state
            state = await ctx.store.get_state()

            # Send an event
            ctx.send_event(ProgressEvent(message=f"Processing {state.name}"))

            # Write to stream
            ctx.write_event_to_stream(
                ProgressEvent(message=f"Counter at {state.counter}")
            )

            # Update state
            await ctx.store.set_state(
                MyState(counter=state.counter + 1, name=state.name)
            )

            # Return an event
            return CustomEvent(count=state.counter)

        @step
        async def final_step(self, ev: CustomEvent) -> StopEvent:
            return StopEvent(result=ev.count)

    workflow = TestWorkflow()
    ctx = TestStepContext(workflow, initial_state=MyState(counter=5, name="test"))

    # Call the step
    result = await workflow.my_step(ctx, StartEvent())

    # Assertions on the result
    assert isinstance(result, CustomEvent)
    assert result.count == 5  # Original counter value

    # Assertions on tracked events
    assert len(ctx.sent_events) == 1
    sent_event, target = ctx.sent_events[0]
    assert isinstance(sent_event, ProgressEvent)
    assert sent_event.message == "Processing test"
    assert target is None

    assert len(ctx.streamed_events) == 1
    assert isinstance(ctx.streamed_events[0], ProgressEvent)
    assert ctx.streamed_events[0].message == "Counter at 5"

    # Assertions on state changes
    new_state = await ctx.store.get_state()
    assert new_state.counter == 6


@pytest.mark.asyncio
async def test_collect_events_raises_not_implemented() -> None:
    """Test that collect_events raises NotImplementedError."""
    ctx = TestStepContext.create()
    with pytest.raises(NotImplementedError, match="collect_events is not supported"):
        ctx.collect_events(CustomEvent(count=1), [CustomEvent])


@pytest.mark.asyncio
async def test_wait_for_event_raises_not_implemented() -> None:
    """Test that wait_for_event raises NotImplementedError."""
    ctx = TestStepContext.create()
    with pytest.raises(NotImplementedError, match="wait_for_event is not supported"):
        await ctx.wait_for_event(CustomEvent)


def test_stream_events_raises_not_implemented() -> None:
    """Test that stream_events raises NotImplementedError."""
    ctx = TestStepContext.create()
    with pytest.raises(NotImplementedError, match="stream_events is not supported"):
        ctx.stream_events()


@pytest.mark.asyncio
async def test_running_steps_raises_not_implemented() -> None:
    """Test that running_steps raises NotImplementedError."""
    ctx = TestStepContext.create()
    with pytest.raises(NotImplementedError, match="running_steps is not supported"):
        await ctx.running_steps()


@pytest.mark.asyncio
async def test_multiple_operations() -> None:
    """Test multiple operations in sequence with clearing."""
    ctx = TestStepContext.create(DictState(value=0))

    # First operation
    ctx.send_event(CustomEvent(count=1))
    ctx.write_event_to_stream(ProgressEvent(message="op1"))
    assert len(ctx.sent_events) == 1
    assert len(ctx.streamed_events) == 1

    # Clear and second operation
    ctx.clear_tracked_events()
    ctx.send_event(CustomEvent(count=2))
    ctx.send_event(CustomEvent(count=3))
    ctx.write_event_to_stream(ProgressEvent(message="op2"))

    # Should only have events from second operation
    assert len(ctx.sent_events) == 2
    assert len(ctx.streamed_events) == 1
    assert ctx.sent_events[0][0].count == 2  # type: ignore
    assert ctx.sent_events[1][0].count == 3  # type: ignore
    assert ctx.streamed_events[0].message == "op2"  # type: ignore


@pytest.mark.asyncio
async def test_is_running_property() -> None:
    """Test that is_running always returns True."""
    ctx = TestStepContext.create()
    assert ctx.is_running is True

    # Even after operations
    ctx.send_event(CustomEvent(count=1))
    assert ctx.is_running is True


@pytest.mark.asyncio
async def test_store_with_dict_state_paths() -> None:
    """Test that store methods work with DictState paths."""
    ctx = TestStepContext.create()

    # Set nested values
    await ctx.store.set("user.name", "Alice")
    await ctx.store.set("user.age", 30)
    await ctx.store.set("config.enabled", True)

    # Get nested values
    name = await ctx.store.get("user.name")
    age = await ctx.store.get("user.age")
    enabled = await ctx.store.get("config.enabled")

    assert name == "Alice"
    assert age == 30
    assert enabled is True
