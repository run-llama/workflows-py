# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import asyncio
import json
from typing import Optional, Union

import pytest
from pydantic import BaseModel
from workflows.context import Context
from workflows.context.context import (
    _warn_cancel_before_start,
    _warn_cancel_in_step,
    _warn_get_result,
    _warn_is_running_in_step,
)
from workflows.context.external_context import ExternalContext
from workflows.context.state_store import DictState, InMemoryStateStore
from workflows.decorators import step
from workflows.errors import ContextStateError, WorkflowRuntimeError
from workflows.events import (
    Event,
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
)
from workflows.plugins.basic import AsyncioAdapterQueues, BasicRuntime, setting_run_id
from workflows.runtime.types.internal_state import BrokerState
from workflows.runtime.types.ticks import TickAddEvent
from workflows.testing import WorkflowTestRunner
from workflows.workflow import Workflow

from ..conftest import (  # type: ignore[import]
    AnotherTestEvent,
    LastEvent,
    OneTestEvent,
)


@pytest.fixture()
def internal_ctx(workflow: Workflow) -> Context:
    """Create a context directly in internal face for testing store operations."""
    # Set up a runtime with state store for this workflow
    runtime = BasicRuntime()
    run_id = "test-run"
    init_state = BrokerState.from_workflow(workflow)
    # Create queues with state store so get_internal_adapter() returns adapter with store
    queues = AsyncioAdapterQueues(
        run_id=run_id,
        init_state=init_state,
        state_store=InMemoryStateStore(DictState()),
    )
    runtime._queues[run_id] = queues
    workflow._runtime = runtime
    with setting_run_id(run_id):
        return Context._create_internal(workflow=workflow)


@pytest.mark.asyncio
async def test_collect_events() -> None:
    ev1 = OneTestEvent()
    ev2 = AnotherTestEvent()

    class TestWorkflow(Workflow):
        @step
        async def step1(self, _: StartEvent) -> OneTestEvent:
            return ev1

        @step
        async def step2(self, _: StartEvent) -> AnotherTestEvent:
            return ev2

        @step
        async def step3(
            self, ctx: Context, ev: Union[OneTestEvent, AnotherTestEvent]
        ) -> Optional[StopEvent]:
            events = ctx.collect_events(ev, [OneTestEvent, AnotherTestEvent])
            if events is None:
                return None
            return StopEvent(result=events)

    r = await WorkflowTestRunner(TestWorkflow()).run()
    assert r.result == [ev1, ev2]


@pytest.mark.asyncio
async def test_collect_events_empty_expected_list() -> None:
    """
    Test that collect_events returns an empty list (not None) when the
    expected list is empty. This edge case should immediately return []
    since there are no events to collect.
    """

    class TestWorkflow(Workflow):
        @step
        async def start_step(self, ctx: Context, ev: StartEvent) -> StopEvent:
            # Pass an empty list of expected events
            events = ctx.collect_events(ev, [])
            # Should return an empty list, not None
            return StopEvent(result=events)

    r = await WorkflowTestRunner(TestWorkflow()).run()
    assert r.result == []


@pytest.mark.asyncio
async def test_collect_events_with_extra_event_type() -> None:
    """
    Test that collect_events properly handles when an event of a different type
    arrives first, before the expected events.

    This validates that when collect_events is called with an event that's NOT
    in the expected types list, it returns None and waits for matching events.
    """

    class TestWorkflow(Workflow):
        @step
        async def start_step(
            self, ctx: Context, ev: StartEvent
        ) -> Union[OneTestEvent, AnotherTestEvent, LastEvent]:
            await ctx.store.set("num_to_collect", 2)
            await ctx.store.set("calls", 0)
            # Send a LastEvent first (not in the expected collection types)
            ctx.send_event(LastEvent())
            # Then send the events we want to collect
            ctx.send_event(OneTestEvent(test_param="first"))
            ctx.send_event(AnotherTestEvent(another_test_param="second"))
            return None  # type: ignore

        @step
        async def collector(
            self, ctx: Context, ev: Union[OneTestEvent, AnotherTestEvent, LastEvent]
        ) -> Optional[StopEvent]:
            # Track how many times this step is called
            calls = await ctx.store.get("calls")
            await ctx.store.set("calls", calls + 1)

            # Try to collect OneTestEvent and AnotherTestEvent
            # LastEvent is NOT in this list
            events = ctx.collect_events(ev, [OneTestEvent, AnotherTestEvent])
            if events is None:
                # This happens when we receive LastEvent or haven't received all events yet
                return None

            # Verify we got the right events
            assert len(events) == 2
            assert isinstance(events[0], OneTestEvent)
            assert isinstance(events[1], AnotherTestEvent)
            assert events[0].test_param == "first"
            assert events[1].another_test_param == "second"

            return StopEvent(result="collected")

    r = await WorkflowTestRunner(TestWorkflow()).run()
    assert r.result == "collected"

    # Verify the collector was called multiple times (once for each event)
    ctx = r.ctx
    assert ctx is not None
    ctx_dict = ctx.to_dict()
    # State is serialized as JSON strings under state_data._data
    calls = json.loads(ctx_dict["state"]["state_data"]["_data"]["calls"])
    # Should be called at least 3 times: once for LastEvent (returns None),
    # once for OneTestEvent (returns None), once for AnotherTestEvent (returns result)
    assert calls >= 3


@pytest.mark.asyncio
async def test_get_default(internal_ctx: Context) -> None:
    assert await internal_ctx.store.get("test_key", default=42) == 42


@pytest.mark.asyncio
async def test_get(internal_ctx: Context) -> None:
    await internal_ctx.store.set("foo", 42)
    assert await internal_ctx.store.get("foo") == 42


@pytest.mark.asyncio
async def test_get_not_found(internal_ctx: Context) -> None:
    with pytest.raises(ValueError):
        await internal_ctx.store.get("foo")


@pytest.mark.asyncio
async def test_send_event_step_is_none(workflow: Workflow) -> None:
    ev = Event(foo="bar")
    # Create a fresh context and run workflow
    ctx = Context(workflow)
    handler = ctx._workflow_run(workflow, start_event=StartEvent())
    try:
        handler.ctx.send_event(ev)
        await asyncio.sleep(0.01)
        # handler.ctx is a new external context
        external_face = handler.ctx._face
        assert isinstance(external_face, ExternalContext)
        replay = external_face._tick_log
        assert TickAddEvent(event=ev, step_name=None) in replay
    finally:
        external_face = handler.ctx._face
        assert isinstance(external_face, ExternalContext)
        await external_face.shutdown()


@pytest.mark.asyncio
async def test_send_event_to_non_existent_step(ctx: Context) -> None:
    with pytest.raises(
        WorkflowRuntimeError, match="Step does_not_exist does not exist"
    ):
        ctx.send_event(Event(), "does_not_exist")


@pytest.mark.asyncio
async def test_send_event_to_wrong_step(ctx: Context) -> None:
    with pytest.raises(
        WorkflowRuntimeError,
        match="Step middle_step does not accept event of type <class 'workflows.events.Event'>",
    ):
        ctx.send_event(Event(), "middle_step")


@pytest.mark.asyncio
async def test_empty_inprogress_when_workflow_done(workflow: Workflow) -> None:
    result = await WorkflowTestRunner(workflow).run()
    ctx = result.ctx

    # there shouldn't be any in progress events
    assert ctx is not None
    assert isinstance(ctx._face, ExternalContext)
    # After workflow completion, in_progress should be empty for all steps
    state = ctx._face._state
    for step_name, worker_state in state.workers.items():
        assert len(worker_state.in_progress) == 0, (
            f"Step {step_name} has {len(worker_state.in_progress)} in-progress events"
        )


@pytest.mark.asyncio
async def test_wait_for_event_in_workflow() -> None:
    class TestWorkflow(Workflow):
        @step
        async def step1(self, ctx: Context, ev: StartEvent) -> StopEvent:
            result = await ctx.wait_for_event(
                Event,
                waiter_event=Event(msg="foo"),
                waiter_id="test_id",
            )
            return StopEvent(result=result.msg)

    workflow = TestWorkflow()
    handler = workflow.run()
    assert handler.ctx
    async for ev in handler.stream_events():
        if isinstance(ev, Event) and ev.msg == "foo":
            handler.ctx.send_event(Event(msg="bar"))
            break

    result = await handler
    assert result == "bar"


class CustomState(BaseModel):
    pass


@pytest.mark.asyncio
async def test_wait_for_event_in_workflow_serialization() -> None:
    """Ensure hitl works with serialization and custom state."""

    class TestWorkflow(Workflow):
        @step
        async def step1(self, ctx: Context[CustomState], ev: StartEvent) -> StopEvent:
            result = await ctx.wait_for_event(
                Event,
                waiter_event=Event(msg="foo"),
                waiter_id="test_id",
            )
            return StopEvent(result=result.msg)

    workflow = TestWorkflow()
    handler = workflow.run()
    ctx_dict = None

    assert handler.ctx
    async for ev in handler.stream_events():
        if isinstance(ev, Event) and ev.msg == "foo":
            ctx_dict = handler.ctx.to_dict()
            # Check that at least one worker has waiters
            assert ctx_dict["version"] == 1
            total_waiters = sum(
                len(worker_data["collected_waiters"])
                for worker_data in ctx_dict["workers"].values()
            )
            assert total_waiters == 1
            await handler.cancel_run()
            break

    # Roundtrip the context
    assert ctx_dict is not None
    # verify creating a new context has the correct state
    new_ctx = Context.from_dict(workflow, ctx_dict)
    new_handler = workflow.run(ctx=new_ctx)
    assert isinstance(new_handler.ctx._face, ExternalContext)
    # Check that the waiters are properly restored
    state = new_handler.ctx._face._state
    total_waiters = sum(
        len(worker.collected_waiters) for worker in state.workers.values()
    )
    assert total_waiters == 1

    # Continue the workflow
    assert new_handler.ctx
    new_handler.ctx.send_event(Event(msg="bar"))
    result = await new_handler
    assert result == "bar"
    assert isinstance(new_handler.ctx._face, ExternalContext)
    # After workflow completion, there should be no more waiters
    state = new_handler.ctx._face._state
    total_waiters = sum(
        len(worker.collected_waiters) for worker in state.workers.values()
    )
    assert total_waiters == 0


class Waiter1(Event):
    msg: str


class Waiter2(Event):
    msg: str


class ResultEvent(Event):
    result: str


class WaitingWorkflow(Workflow):
    @step
    async def spawn_waiters(
        self, ctx: Context, ev: StartEvent
    ) -> Union[Waiter1, Waiter2]:
        ctx.send_event(Waiter1(msg="foo"))
        ctx.send_event(Waiter2(msg="bar"))
        return None  # type: ignore

    @step
    async def waiter_one(self, ctx: Context, ev: Waiter1) -> ResultEvent:
        ctx.write_event_to_stream(InputRequiredEvent(prefix="waiter_one"))  # type: ignore

        new_ev: HumanResponseEvent = await ctx.wait_for_event(
            HumanResponseEvent,
            requirements={"waiter_id": "waiter_one"},
        )
        return ResultEvent(result=new_ev.response)

    @step
    async def waiter_two(self, ctx: Context, ev: Waiter2) -> ResultEvent:
        ctx.write_event_to_stream(InputRequiredEvent(prefix="waiter_two"))  # type: ignore

        new_ev: HumanResponseEvent = await ctx.wait_for_event(
            HumanResponseEvent,
            requirements={"waiter_id": "waiter_two"},
        )
        return ResultEvent(result=new_ev.response)

    @step
    async def collect_waiters(self, ctx: Context, ev: ResultEvent) -> StopEvent:
        events: list[ResultEvent] | None = ctx.collect_events(  # type: ignore
            ev, [ResultEvent, ResultEvent]
        )
        if events is None:
            return None  # type: ignore

        return StopEvent(result=[e.result for e in events])


@pytest.mark.asyncio
async def test_wait_for_multiple_events_in_workflow() -> None:
    workflow = WaitingWorkflow()
    handler = workflow.run()
    assert handler.ctx

    async for ev in handler.stream_events():
        if isinstance(ev, InputRequiredEvent) and ev.prefix == "waiter_one":
            handler.ctx.send_event(
                HumanResponseEvent(response="foo", waiter_id="waiter_one")  # type: ignore
            )
        elif isinstance(ev, InputRequiredEvent) and ev.prefix == "waiter_two":
            handler.ctx.send_event(
                HumanResponseEvent(response="bar", waiter_id="waiter_two")  # type: ignore
            )

    result = await handler
    assert result == ["foo", "bar"]
    assert not handler.ctx.is_running

    # serialize and resume
    ctx_dict = handler.ctx.to_dict()
    ctx = Context.from_dict(workflow, ctx_dict)
    handler = workflow.run(ctx=ctx)
    assert handler.ctx

    async for ev in handler.stream_events():
        if isinstance(ev, InputRequiredEvent) and ev.prefix == "waiter_one":
            handler.ctx.send_event(
                HumanResponseEvent(response="fizz", waiter_id="waiter_one")  # type: ignore
            )
        elif isinstance(ev, InputRequiredEvent) and ev.prefix == "waiter_two":
            handler.ctx.send_event(
                HumanResponseEvent(response="buzz", waiter_id="waiter_two")  # type: ignore
            )

    result = await handler
    assert result == ["fizz", "buzz"]


@pytest.mark.asyncio
async def test_clear(internal_ctx: Context) -> None:
    await internal_ctx.store.set("test_key", 42)
    await internal_ctx.store.clear()
    res = await internal_ctx.store.get("test_key", default=None)
    assert res is None


@pytest.mark.asyncio
async def test_running_steps_before_run_raises(workflow: Workflow) -> None:
    """Calling running_steps() before workflow.run() should raise ContextStateError."""
    ctx = Context(workflow)
    with pytest.raises(ContextStateError, match="requires a running workflow"):
        await ctx.running_steps()


@pytest.mark.asyncio
async def test_store_access_outside_step_works() -> None:
    """Accessing ctx.store from handler code (outside step) should work."""

    class SimpleWorkflow(Workflow):
        @step
        async def only(self, ev: StartEvent) -> StopEvent:
            return StopEvent(result="done")

    wf = SimpleWorkflow()
    handler = wf.run()

    # Access store from external context (handler code) should work
    store = handler.ctx.store
    assert isinstance(store, InMemoryStateStore)

    # Verify reads and writes work
    await store.set("key", "value")
    assert await store.get("key") == "value"

    await handler


@pytest.mark.asyncio
async def test_store_access_before_run_works() -> None:
    """Accessing ctx.store before workflow.run() should return a staging store."""

    class SimpleWorkflow(Workflow):
        @step
        async def only(self, ev: StartEvent) -> StopEvent:
            return StopEvent(result="done")

    wf = SimpleWorkflow()
    ctx = Context(wf)

    store = ctx.store
    assert isinstance(store, InMemoryStateStore)
    await store.set("key", "value")
    assert await store.get("key") == "value"


@pytest.mark.asyncio
async def test_store_seed_before_run_visible_in_step() -> None:
    """State seeded via ctx.store before run should be visible inside steps."""

    class SeededWorkflow(Workflow):
        @step
        async def read_seed(self, ctx: Context, ev: StartEvent) -> StopEvent:
            val = await ctx.store.get("seeded_key")
            return StopEvent(result=val)

    wf = SeededWorkflow()
    ctx = Context(wf)
    await ctx.store.set("seeded_key", "hello")

    result = await wf.run(ctx=ctx)
    assert result == "hello"


@pytest.mark.asyncio
async def test_store_seed_on_deserialized_context() -> None:
    """Seeding state on a deserialized context should merge with existing state."""

    class StatefulWorkflow(Workflow):
        @step
        async def first_step(self, ctx: Context, ev: StartEvent) -> StopEvent:
            # Just pass through; state is set externally
            return StopEvent(result="done")

    class ReadWorkflow(Workflow):
        @step
        async def check(self, ctx: Context, ev: StartEvent) -> StopEvent:
            old = await ctx.store.get("existing")
            new = await ctx.store.get("added")
            return StopEvent(result=f"{old}-{new}")

    wf = StatefulWorkflow()

    # First run to create some state
    ctx = Context(wf)
    await ctx.store.set("existing", "orig")
    handler = wf.run(ctx=ctx)
    await handler

    # Serialize and restore (use ReadWorkflow for second run)
    ctx_dict = ctx.to_dict()
    read_wf = ReadWorkflow()
    restored_ctx = Context.from_dict(read_wf, ctx_dict)

    # Seed additional state before next run
    await restored_ctx.store.set("added", "new")

    result = await read_wf.run(ctx=restored_ctx)
    assert result == "orig-new"


@pytest.mark.asyncio
async def test_store_continuation_with_pre_run_seeding() -> None:
    """Continuation runs with pre-run seeding should carry state through."""

    class CountWorkflow(Workflow):
        @step
        async def increment(self, ctx: Context, ev: StartEvent) -> StopEvent:
            count = await ctx.store.get("count", default=0)
            count += 1
            await ctx.store.set("count", count)
            return StopEvent(result=count)

    wf = CountWorkflow()
    ctx = Context(wf)

    # Seed initial count
    await ctx.store.set("count", 10)

    # First run: 10 + 1 = 11
    result = await wf.run(ctx=ctx)
    assert result == 11

    # Second run (continuation): 11 + 1 = 12
    result = await wf.run(ctx=ctx)
    assert result == 12


@pytest.mark.asyncio
async def test_to_dict_before_run_raises(workflow: Workflow) -> None:
    """Calling to_dict() before workflow.run() should raise ContextStateError."""
    ctx = Context(workflow)
    with pytest.raises(ContextStateError, match="requires a running workflow"):
        ctx.to_dict()


@pytest.mark.asyncio
async def test_stream_events_before_run_raises(workflow: Workflow) -> None:
    """Calling stream_events() before workflow.run() should raise ContextStateError."""
    ctx = Context(workflow)
    with pytest.raises(ContextStateError, match="requires a running workflow"):
        ctx.stream_events()


# ============================================================================
# Warning Tests
# ============================================================================


@pytest.mark.asyncio
async def test_cancel_before_start_warns(workflow: Workflow) -> None:
    """Calling cancel before run() should emit warning."""
    # Clear the lru_cache to ensure warning fires
    _warn_cancel_before_start.cache_clear()

    ctx = Context(workflow)
    with pytest.warns(UserWarning, match="cancel.*called before workflow started"):
        ctx._workflow_cancel_run()


@pytest.mark.asyncio
async def test_send_event_before_start_raises(workflow: Workflow) -> None:
    """Sending event before run() should raise ContextStateError."""
    ctx = Context(workflow)
    with pytest.raises(
        ContextStateError, match="send_event.*called before workflow started"
    ):
        ctx.send_event(Event())


@pytest.mark.asyncio
async def test_is_running_in_step_warns() -> None:
    """Calling is_running from within a step should emit deprecation warning."""
    # Clear the lru_cache to ensure warning fires
    _warn_is_running_in_step.cache_clear()

    is_running_value = None

    class TestWorkflow(Workflow):
        @step
        async def check_running(self, ctx: Context, ev: StartEvent) -> StopEvent:
            nonlocal is_running_value
            is_running_value = ctx.is_running
            return StopEvent(result="done")

    wf = TestWorkflow()
    with pytest.warns(DeprecationWarning, match="is_running called from within a step"):
        await wf.run()

    # Should still return True despite the warning
    assert is_running_value is True


@pytest.mark.asyncio
async def test_cancel_in_step_warns() -> None:
    """Calling cancel from within a step should emit warning."""
    # Clear the lru_cache to ensure warning fires
    _warn_cancel_in_step.cache_clear()

    class TestWorkflow(Workflow):
        @step
        async def cancel_self(self, ctx: Context, ev: StartEvent) -> StopEvent:
            ctx._workflow_cancel_run()
            return StopEvent(result="done")

    wf = TestWorkflow()
    with pytest.warns(UserWarning, match="cancel.*called from within a step"):
        await wf.run()


@pytest.mark.asyncio
async def test_get_result_before_complete_raises() -> None:
    """Calling get_result() while workflow still running should raise WorkflowRuntimeError."""
    # Clear the lru_cache to ensure deprecation warning fires
    _warn_get_result.cache_clear()

    step_started = asyncio.Event()
    step_continue = asyncio.Event()

    class SlowWorkflow(Workflow):
        @step
        async def slow(self, ev: StartEvent) -> StopEvent:
            step_started.set()
            await step_continue.wait()
            return StopEvent(result="done")

    wf = SlowWorkflow()
    handler = wf.run()

    # Wait for step to start
    await step_started.wait()

    # Try to get result before workflow completes - should raise
    with pytest.warns(DeprecationWarning):  # get_result is deprecated
        with pytest.raises(WorkflowRuntimeError, match="is not complete"):
            handler.ctx.get_result()

    # Let workflow complete
    step_continue.set()
    await handler


@pytest.mark.asyncio
async def test_get_result_pre_context_raises(workflow: Workflow) -> None:
    """Calling get_result() before run() should raise ContextStateError."""
    # Clear the lru_cache to ensure deprecation warning fires
    _warn_get_result.cache_clear()

    ctx = Context(workflow)

    with pytest.warns(DeprecationWarning):  # get_result is deprecated
        with pytest.raises(ContextStateError, match="requires a running workflow"):
            ctx.get_result()
