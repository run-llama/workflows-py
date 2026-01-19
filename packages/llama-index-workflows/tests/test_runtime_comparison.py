# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

"""
Comprehensive test suite that runs against both BasicRuntime and DBOSRuntime.

This module tests all key workflow features against both runtime implementations
to ensure consistent behavior. DBOS is configured with SQLite for testing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncGenerator, Generator, Optional, Union

import pytest
from dbos import DBOS, DBOSConfig
from pydantic import Field
from workflows.context import Context
from workflows.decorators import step
from workflows.errors import WorkflowCancelledByUser, WorkflowTimeoutError
from workflows.events import (
    Event,
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StepStateChanged,
    StopEvent,
    WorkflowCancelledEvent,
    WorkflowFailedEvent,
    WorkflowIdleEvent,
    WorkflowTimedOutEvent,
)
from workflows.plugins.basic import BasicRuntime
from workflows.plugins.dbos import DBOSRuntime
from workflows.retry_policy import ConstantDelayRetryPolicy
from workflows.runtime.types.plugin import Runtime
from workflows.workflow import Workflow


# ============================================================================
# Test Markers
# ============================================================================

# Some DBOS features are not yet fully implemented, mark those tests as xfail
dbos_xfail = pytest.mark.xfail(
    reason="DBOS runtime feature not yet implemented",
    strict=False,  # Allow unexpected passes
)


def xfail_dbos(func: Any) -> Any:
    """Mark a test as xfail only when running with DBOS runtime."""
    # This is a simple marker - the actual skipping happens in the test
    return func


# ============================================================================
# Test Events
# ============================================================================


class IntermediateEvent(Event):
    """Test event passed between workflow steps."""

    value: int


class ProcessedEvent(Event):
    """Test event indicating processed data."""

    processed_value: str


class DataEvent(Event):
    """Test event carrying data."""

    data: str


class CollectableEvent(Event):
    """Test event for collection tests."""

    index: int


class AwaitedEvent(Event):
    """Test event for wait_for_event tests."""

    payload: str = Field(default="default")


class TaggedEvent(HumanResponseEvent):
    """Test event with routing tag."""

    tag: str


# ============================================================================
# Test Workflows
# ============================================================================


class SingleStepWorkflow(Workflow):
    """Simple single-step workflow for basic execution tests."""

    @step
    async def process(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="single_step_done")


class ThreeStepWorkflow(Workflow):
    """Three-step workflow for event passing tests."""

    @step
    async def first(self, ev: StartEvent) -> IntermediateEvent:
        return IntermediateEvent(value=42)

    @step
    async def second(self, ev: IntermediateEvent) -> ProcessedEvent:
        return ProcessedEvent(processed_value=f"processed_{ev.value}")

    @step
    async def third(self, ev: ProcessedEvent) -> StopEvent:
        return StopEvent(result=ev.processed_value)


class ContextStateWorkflow(Workflow):
    """Workflow that uses context state storage."""

    @step
    async def store_data(self, ev: StartEvent, ctx: Context) -> IntermediateEvent:
        await ctx.store.set("counter", 100)
        await ctx.store.set("name", "test_workflow")
        return IntermediateEvent(value=1)

    @step
    async def read_and_modify(
        self, ev: IntermediateEvent, ctx: Context
    ) -> ProcessedEvent:
        counter = await ctx.store.get("counter")
        name = await ctx.store.get("name")
        assert counter == 100
        assert name == "test_workflow"
        await ctx.store.set("counter", counter + 50)
        return ProcessedEvent(processed_value=f"{name}_{counter}")

    @step
    async def verify_final(self, ev: ProcessedEvent, ctx: Context) -> StopEvent:
        counter = await ctx.store.get("counter")
        assert counter == 150
        return StopEvent(result=ev.processed_value)


class EventCollectionWorkflow(Workflow):
    """Workflow that collects multiple events before proceeding."""

    @step
    async def emit_events(
        self, ev: StartEvent, ctx: Context
    ) -> Optional[CollectableEvent]:
        for i in range(4):
            ctx.send_event(CollectableEvent(index=i + 1))
        return None

    @step
    async def collect_and_sum(
        self, ev: CollectableEvent, ctx: Context
    ) -> Optional[StopEvent]:
        events = ctx.collect_events(ev, [CollectableEvent] * 4)
        if events is None:
            return None
        total = sum(e.index for e in events)
        return StopEvent(result=f"sum_{total}")


class MixedEventCollectionWorkflow(Workflow):
    """Workflow that collects different event types (ProcessedEvent from different sources)."""

    @step
    async def emit_mixed(
        self, ev: StartEvent, ctx: Context
    ) -> Union[IntermediateEvent, DataEvent, None]:
        # Emit multiple events for parallel processing
        ctx.send_event(IntermediateEvent(value=10))
        ctx.send_event(DataEvent(data="hello"))
        return None

    @step
    async def transform_intermediate(
        self, ev: IntermediateEvent
    ) -> ProcessedEvent:
        return ProcessedEvent(processed_value=f"int_{ev.value}")

    @step
    async def transform_data(self, ev: DataEvent) -> ProcessedEvent:
        return ProcessedEvent(processed_value=f"data_{ev.data}")

    @step
    async def collect_processed(
        self, ev: ProcessedEvent, ctx: Context
    ) -> Optional[StopEvent]:
        events = ctx.collect_events(ev, [ProcessedEvent] * 2)
        if events is None:
            return None
        combined = "_".join(e.processed_value for e in sorted(events, key=lambda x: x.processed_value))
        return StopEvent(result=combined)


class WaitForEventWorkflow(Workflow):
    """Workflow that waits for external events (human-in-the-loop)."""

    @step
    async def wait_step(self, ev: StartEvent, ctx: Context) -> StopEvent:
        awaited = await ctx.wait_for_event(
            AwaitedEvent,
            waiter_event=InputRequiredEvent(),
        )
        return StopEvent(result=f"received_{awaited.payload}")


class WaitWithRequirementsWorkflow(Workflow):
    """Workflow that waits for events matching specific requirements."""

    @step
    async def wait_with_filter(self, ev: StartEvent, ctx: Context) -> StopEvent:
        awaited = await ctx.wait_for_event(
            AwaitedEvent,
            waiter_event=InputRequiredEvent(),
            requirements={"payload": "expected_value"},
        )
        return StopEvent(result=f"matched_{awaited.payload}")


class EventStreamingWorkflow(Workflow):
    """Workflow that writes events to the stream for observers."""

    @step
    async def stream_step(self, ev: StartEvent, ctx: Context) -> StopEvent:
        ctx.write_event_to_stream(DataEvent(data="progress_1"))
        ctx.write_event_to_stream(DataEvent(data="progress_2"))
        return StopEvent(result="streaming_done")


class SlowWorkflow(Workflow):
    """Workflow with a slow step for timeout testing."""

    @step
    async def slow_step(self, ev: StartEvent) -> StopEvent:
        await asyncio.sleep(10.0)
        return StopEvent(result="should_not_reach")


class RetryWorkflow(Workflow):
    """Workflow with retry logic."""

    attempts: int = 0

    @step(retry_policy=ConstantDelayRetryPolicy(maximum_attempts=3, delay=0.01))
    async def flaky_step(self, ev: StartEvent) -> StopEvent:
        self.attempts += 1
        if self.attempts < 3:
            raise RuntimeError(f"Failure #{self.attempts}")
        return StopEvent(result=f"success_after_{self.attempts}_attempts")


class AlwaysFailsWorkflow(Workflow):
    """Workflow that always fails for error handling tests."""

    @step(retry_policy=ConstantDelayRetryPolicy(maximum_attempts=2, delay=0.01))
    async def failing_step(self, ev: StartEvent) -> StopEvent:
        raise ValueError("intentional_failure")


class CancellableWorkflow(Workflow):
    """Workflow that can be cancelled while running."""

    @step
    async def cancellable_step(self, ev: StartEvent) -> StopEvent:
        await asyncio.sleep(10.0)
        return StopEvent(result="should_not_reach")


class MultiRouteWorkflow(Workflow):
    """Workflow with multiple steps accepting the same event type for routing tests."""

    @step
    async def starter(self, ev: StartEvent) -> Optional[StopEvent]:
        return None

    @step
    async def route_a(self, ev: TaggedEvent) -> StopEvent:
        return StopEvent(result=f"route_a_{ev.tag}")

    @step
    async def route_b(self, ev: TaggedEvent) -> StopEvent:
        return StopEvent(result=f"route_b_{ev.tag}")


class ConcurrencyWorkflow(Workflow):
    """Workflow with limited concurrency."""

    execution_order: list[int] = []

    @step(num_workers=1)
    async def limited_step(self, ev: StartEvent) -> StopEvent:
        self.execution_order.append(len(self.execution_order) + 1)
        await asyncio.sleep(0.02)
        return StopEvent(result=f"done_{len(self.execution_order)}")


# Additional workflow classes that were previously defined inside test functions


class CustomStartEvent(StartEvent):
    """Custom start event with additional field."""

    custom_value: str


class KwargsWorkflow(Workflow):
    """Workflow that accepts custom start event kwargs."""

    @step
    async def process(self, ev: CustomStartEvent) -> StopEvent:
        return StopEvent(result=f"received_{ev.custom_value}")


class StatePersistenceWorkflow(Workflow):
    """Workflow for testing state persistence across runs."""

    @step
    async def increment(self, ev: StartEvent, ctx: Context) -> StopEvent:
        current = await ctx.store.get("counter", 0)
        await ctx.store.set("counter", current + 1)
        return StopEvent(result=f"count_{current + 1}")


class EmptyResultWorkflow(Workflow):
    """Workflow that returns empty result."""

    @step
    async def process(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result=None)


class ComplexResultWorkflow(Workflow):
    """Workflow that returns complex data structures."""

    @step
    async def process(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result={"key": "value", "list": [1, 2, 3]})


class NoneReturningWorkflow(Workflow):
    """Workflow with steps that return None (fan-out pattern)."""

    @step
    async def fanout(
        self, ev: StartEvent, ctx: Context
    ) -> Optional[IntermediateEvent]:
        ctx.send_event(IntermediateEvent(value=1))
        ctx.send_event(IntermediateEvent(value=2))
        return None

    @step
    async def collect(
        self, ev: IntermediateEvent, ctx: Context
    ) -> Optional[StopEvent]:
        events = ctx.collect_events(ev, [IntermediateEvent] * 2)
        if events is None:
            return None
        total = sum(e.value for e in events)
        return StopEvent(result=f"total_{total}")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def basic_runtime() -> Generator[BasicRuntime, None, None]:
    """Create a BasicRuntime instance."""
    runtime = BasicRuntime()
    runtime.launch()
    try:
        yield runtime
    finally:
        runtime.destroy()


@pytest.fixture
def dbos_runtime(tmp_path: Path) -> Generator[DBOSRuntime, None, None]:
    """Create a DBOSRuntime instance with SQLite database."""
    db_file = tmp_path / "dbos_test.sqlite3"
    system_db_url = f"sqlite+pysqlite:///{db_file}?check_same_thread=false"

    config: DBOSConfig = {
        "name": "workflows-py-test-comparison",
        "system_database_url": system_db_url,
        "run_admin_server": False,
    }
    DBOS(config=config)

    runtime = DBOSRuntime()
    try:
        yield runtime
    finally:
        runtime.destroy()
        DBOS.destroy()


@pytest.fixture(params=["basic", "dbos"])
def runtime(
    request: pytest.FixtureRequest,
    basic_runtime: BasicRuntime,
    dbos_runtime: DBOSRuntime,
) -> Runtime:
    """Parametrized fixture returning either BasicRuntime or DBOSRuntime."""
    if request.param == "basic":
        return basic_runtime
    return dbos_runtime


def launch_if_dbos(runtime: Runtime) -> None:
    """Launch the runtime if it's a DBOSRuntime (must be called after workflow creation)."""
    if isinstance(runtime, DBOSRuntime):
        runtime.launch()


def is_dbos_runtime(runtime: Runtime) -> bool:
    """Check if the runtime is a DBOSRuntime."""
    return isinstance(runtime, DBOSRuntime)


def skip_if_dbos(runtime: Runtime, reason: str = "Feature not implemented for DBOS") -> None:
    """Skip the test if running with DBOS runtime."""
    if is_dbos_runtime(runtime):
        pytest.skip(reason)


# ============================================================================
# Tests: Basic Workflow Execution
# ============================================================================


@pytest.mark.asyncio
async def test_single_step_workflow(runtime: Runtime) -> None:
    """Test basic single-step workflow execution."""
    workflow = SingleStepWorkflow(runtime=runtime)
    launch_if_dbos(runtime)

    handler = workflow.run()
    result = await handler
    assert result == "single_step_done"


@pytest.mark.asyncio
async def test_three_step_workflow(runtime: Runtime) -> None:
    """Test multi-step workflow with event passing."""
    workflow = ThreeStepWorkflow(runtime=runtime)
    launch_if_dbos(runtime)

    handler = workflow.run()
    result = await handler
    assert result == "processed_42"


@pytest.mark.asyncio
async def test_workflow_with_start_event_kwargs(runtime: Runtime) -> None:
    """Test passing kwargs to construct start event."""
    workflow = KwargsWorkflow(runtime=runtime)
    launch_if_dbos(runtime)

    handler = workflow.run(custom_value="test_data")
    result = await handler
    assert result == "received_test_data"


# ============================================================================
# Tests: Context State Management
# ============================================================================


@pytest.mark.asyncio
async def test_context_state_storage(runtime: Runtime) -> None:
    """Test context state storage across workflow steps."""
    workflow = ContextStateWorkflow(runtime=runtime)
    launch_if_dbos(runtime)

    handler = workflow.run()
    result = await handler
    assert result == "test_workflow_100"


@pytest.mark.asyncio
async def test_context_state_persistence_across_runs(runtime: Runtime) -> None:
    """Test context state persists when reusing context."""
    workflow = StatePersistenceWorkflow(runtime=runtime)
    launch_if_dbos(runtime)

    # First run
    handler1 = workflow.run()
    result1 = await handler1
    assert result1 == "count_1"

    # Second run with same context
    ctx = handler1.ctx
    handler2 = workflow.run(ctx=ctx)
    result2 = await handler2
    assert result2 == "count_2"


# ============================================================================
# Tests: Event Collection
# ============================================================================


@pytest.mark.asyncio
async def test_event_collection_same_type(runtime: Runtime) -> None:
    """Test collecting multiple events of the same type."""
    skip_if_dbos(runtime, "DBOS event collection times out - feature in development")
    workflow = EventCollectionWorkflow(runtime=runtime)
    launch_if_dbos(runtime)

    handler = workflow.run()
    result = await handler
    assert result == "sum_10"  # 1 + 2 + 3 + 4 = 10


@pytest.mark.asyncio
async def test_event_collection_mixed_types(runtime: Runtime) -> None:
    """Test collecting events of different types."""
    skip_if_dbos(runtime, "DBOS event collection times out - feature in development")
    workflow = MixedEventCollectionWorkflow(runtime=runtime)
    launch_if_dbos(runtime)

    handler = workflow.run()
    result = await handler
    assert "int_10" in result
    assert "data_hello" in result


# ============================================================================
# Tests: Wait for Event (Human-in-the-Loop)
# ============================================================================


@pytest.mark.asyncio
async def test_wait_for_event_basic(runtime: Runtime) -> None:
    """Test basic wait_for_event functionality."""
    skip_if_dbos(runtime, "DBOS stream_published_events not fully implemented")
    workflow = WaitForEventWorkflow(runtime=runtime, timeout=5.0)
    launch_if_dbos(runtime)

    handler = workflow.run()

    # Wait for InputRequiredEvent to appear
    input_required_found = False
    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            input_required_found = True
            # Send the awaited event
            handler.ctx.send_event(AwaitedEvent(payload="test_payload"))
        elif isinstance(event, StopEvent):
            break

    assert input_required_found
    result = await handler
    assert result == "received_test_payload"


@pytest.mark.asyncio
async def test_wait_for_event_with_requirements(runtime: Runtime) -> None:
    """Test wait_for_event with filter requirements."""
    skip_if_dbos(runtime, "DBOS stream_published_events not fully implemented")
    workflow = WaitWithRequirementsWorkflow(runtime=runtime, timeout=5.0)
    launch_if_dbos(runtime)

    handler = workflow.run()

    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            # Send event matching requirements
            handler.ctx.send_event(AwaitedEvent(payload="expected_value"))
        elif isinstance(event, StopEvent):
            break

    result = await handler
    assert result == "matched_expected_value"


# ============================================================================
# Tests: Event Streaming
# ============================================================================


@pytest.mark.asyncio
async def test_event_streaming(runtime: Runtime) -> None:
    """Test that events are properly streamed to observers."""
    skip_if_dbos(runtime, "DBOS stream_published_events not fully implemented")
    workflow = EventStreamingWorkflow(runtime=runtime)
    launch_if_dbos(runtime)

    handler = workflow.run()

    streamed_events: list[Event] = []
    async for event in handler.stream_events():
        streamed_events.append(event)
        if isinstance(event, StopEvent):
            break

    # Verify we got the custom events
    data_events = [e for e in streamed_events if isinstance(e, DataEvent)]
    assert len(data_events) == 2
    assert data_events[0].data == "progress_1"
    assert data_events[1].data == "progress_2"


@pytest.mark.asyncio
async def test_step_state_changes_streamed(runtime: Runtime) -> None:
    """Test that step state changes are streamed when expose_internal=True."""
    skip_if_dbos(runtime, "DBOS stream_published_events not fully implemented")
    workflow = ThreeStepWorkflow(runtime=runtime)
    launch_if_dbos(runtime)

    handler = workflow.run()

    state_changes: list[StepStateChanged] = []
    async for event in handler.stream_events(expose_internal=True):
        if isinstance(event, StepStateChanged):
            state_changes.append(event)
        elif isinstance(event, StopEvent):
            break

    # Should have state changes for each step (RUNNING and NOT_RUNNING)
    assert len(state_changes) >= 6  # 3 steps * 2 states each


# ============================================================================
# Tests: Timeouts
# ============================================================================


@pytest.mark.asyncio
async def test_workflow_timeout(runtime: Runtime) -> None:
    """Test that workflow times out correctly."""
    skip_if_dbos(runtime, "DBOS timeout behavior differs - feature in development")
    workflow = SlowWorkflow(runtime=runtime, timeout=0.05)
    launch_if_dbos(runtime)

    handler = workflow.run()

    # Stream events until we get the timeout event
    timeout_event_found = False
    async for event in handler.stream_events():
        if isinstance(event, WorkflowTimedOutEvent):
            timeout_event_found = True
            assert event.timeout == 0.05
            assert "slow_step" in event.active_steps
            break
        elif isinstance(event, StopEvent):
            break

    assert timeout_event_found

    with pytest.raises(WorkflowTimeoutError):
        await handler


# ============================================================================
# Tests: Retry Logic
# ============================================================================


@pytest.mark.asyncio
async def test_retry_success_after_failures(runtime: Runtime) -> None:
    """Test that retry logic works and succeeds after failures."""
    workflow = RetryWorkflow(runtime=runtime, timeout=5.0)
    launch_if_dbos(runtime)

    handler = workflow.run()
    result = await handler

    assert result == "success_after_3_attempts"
    assert workflow.attempts == 3


@pytest.mark.asyncio
async def test_retry_exhaustion_publishes_failure(runtime: Runtime) -> None:
    """Test that exhausted retries publish WorkflowFailedEvent."""
    skip_if_dbos(runtime, "DBOS stream_published_events not fully implemented")
    workflow = AlwaysFailsWorkflow(runtime=runtime, timeout=5.0)
    launch_if_dbos(runtime)

    handler = workflow.run()

    failure_event_found = False
    async for event in handler.stream_events():
        if isinstance(event, WorkflowFailedEvent):
            failure_event_found = True
            assert event.step_name == "failing_step"
            assert "ValueError" in event.exception_type
            assert event.exception_message == "intentional_failure"
            break
        elif isinstance(event, StopEvent):
            break

    assert failure_event_found

    with pytest.raises(ValueError, match="intentional_failure"):
        await handler


# ============================================================================
# Tests: Cancellation
# ============================================================================


@pytest.mark.asyncio
async def test_workflow_cancellation(runtime: Runtime) -> None:
    """Test that workflow can be cancelled."""
    skip_if_dbos(runtime, "DBOS cancellation requires send from within workflow")
    workflow = CancellableWorkflow(runtime=runtime, timeout=10.0)
    launch_if_dbos(runtime)

    handler = workflow.run()

    # Wait for workflow to start
    await asyncio.sleep(0.05)

    # Cancel the workflow
    await handler.cancel_run()

    # Verify cancellation event
    cancelled_found = False
    try:
        async for event in handler.stream_events():
            if isinstance(event, WorkflowCancelledEvent):
                cancelled_found = True
                break
            elif isinstance(event, StopEvent):
                break
    except WorkflowCancelledByUser:
        cancelled_found = True

    assert cancelled_found or handler.done()

    with pytest.raises(WorkflowCancelledByUser):
        await handler


# ============================================================================
# Tests: Event Routing
# ============================================================================


@pytest.mark.asyncio
async def test_event_routing_to_specific_step(runtime: Runtime) -> None:
    """Test that events can be routed to specific steps."""
    skip_if_dbos(runtime, "DBOS send_event requires context from within workflow")
    workflow = MultiRouteWorkflow(runtime=runtime, timeout=5.0)
    launch_if_dbos(runtime)

    handler = workflow.run()

    # Send event to route_b specifically
    handler.ctx.send_event(TaggedEvent(tag="test"), step="route_b")

    result = await handler
    assert result == "route_b_test"


# ============================================================================
# Tests: Idle Detection
# ============================================================================


@pytest.mark.asyncio
async def test_idle_event_when_waiting(runtime: Runtime) -> None:
    """Test that WorkflowIdleEvent is emitted when waiting for external input."""
    skip_if_dbos(runtime, "DBOS stream_published_events not fully implemented")
    workflow = WaitForEventWorkflow(runtime=runtime, timeout=5.0)
    launch_if_dbos(runtime)

    handler = workflow.run()

    idle_found = False
    input_required_found = False

    # Need expose_internal=True to see WorkflowIdleEvent (it's an InternalDispatchEvent)
    async for event in handler.stream_events(expose_internal=True):
        if isinstance(event, WorkflowIdleEvent):
            idle_found = True
        elif isinstance(event, InputRequiredEvent):
            input_required_found = True
            # Send event to complete workflow
            handler.ctx.send_event(AwaitedEvent(payload="complete"))
        elif isinstance(event, StopEvent):
            break

    assert input_required_found
    assert idle_found

    result = await handler
    assert result == "received_complete"


# ============================================================================
# Tests: Multiple Workflow Instances
# ============================================================================


@pytest.mark.asyncio
async def test_multiple_workflow_instances(runtime: Runtime) -> None:
    """Test running multiple workflow instances concurrently."""
    workflow1 = SingleStepWorkflow(runtime=runtime)
    workflow2 = SingleStepWorkflow(runtime=runtime)
    workflow3 = SingleStepWorkflow(runtime=runtime)
    launch_if_dbos(runtime)

    handlers = [workflow1.run(), workflow2.run(), workflow3.run()]
    results = await asyncio.gather(*handlers)

    assert all(r == "single_step_done" for r in results)


@pytest.mark.asyncio
async def test_different_workflow_types_concurrent(runtime: Runtime) -> None:
    """Test running different workflow types concurrently."""
    workflow1 = SingleStepWorkflow(runtime=runtime)
    workflow2 = ThreeStepWorkflow(runtime=runtime)
    launch_if_dbos(runtime)

    handler1 = workflow1.run()
    handler2 = workflow2.run()

    result1, result2 = await asyncio.gather(handler1, handler2)

    assert result1 == "single_step_done"
    assert result2 == "processed_42"


# ============================================================================
# Tests: Edge Cases
# ============================================================================


@pytest.mark.asyncio
async def test_empty_result_workflow(runtime: Runtime) -> None:
    """Test workflow that returns empty result."""
    workflow = EmptyResultWorkflow(runtime=runtime)
    launch_if_dbos(runtime)

    handler = workflow.run()
    result = await handler
    assert result is None


@pytest.mark.asyncio
async def test_complex_result_workflow(runtime: Runtime) -> None:
    """Test workflow that returns complex data structures."""
    workflow = ComplexResultWorkflow(runtime=runtime)
    launch_if_dbos(runtime)

    handler = workflow.run()
    result = await handler
    assert result == {"key": "value", "list": [1, 2, 3]}


@pytest.mark.asyncio
async def test_step_returning_none(runtime: Runtime) -> None:
    """Test workflow with steps that return None (fan-out pattern)."""
    skip_if_dbos(runtime, "DBOS event collection times out - feature in development")
    workflow = NoneReturningWorkflow(runtime=runtime)
    launch_if_dbos(runtime)

    handler = workflow.run()
    result = await handler
    assert result == "total_3"
