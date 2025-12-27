"""Tests for nested workflow composition (include_workflow)."""

from typing import Any

import pytest
from pydantic import BaseModel, Field
from workflows.errors import WorkflowValidationError
from workflows.events import Event, StartEvent, StopEvent

from workflows import Context, Workflow, step

# ============================================================================
# Simple nested workflow tests
# ============================================================================


class ChildStartEvent(StartEvent):
    """Start event for child workflow."""

    value: int


class ChildStopEvent(StopEvent):  # type: ignore
    """Stop event for child workflow."""

    final: int


class ParentTriggerEvent(Event):
    """Event that triggers child workflow."""

    value: int


class ParentCompleteEvent(Event):
    """Event emitted when child completes."""

    result: int


class ParentFinalEvent(StopEvent):
    """Final stop event for parent."""

    final_result: int


class SimpleChildWorkflow(Workflow):
    """A simple child workflow that doubles a number."""

    @step
    async def process(self, ev: ChildStartEvent) -> ChildStopEvent:
        """Double the input value."""
        return ChildStopEvent(final=ev.value * 2)


class SimpleParentWorkflow(Workflow):
    """A parent workflow that includes the child workflow."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.include_workflow(
            SimpleChildWorkflow,
            start_event=ParentTriggerEvent,
            stop_event=ParentCompleteEvent,
            namespace="child",
        )

    @step
    async def start(self, ev: StartEvent) -> ParentTriggerEvent:
        """Start and trigger child."""
        return ParentTriggerEvent(value=5)

    @step
    async def finalize(self, ev: ParentCompleteEvent) -> ParentFinalEvent:
        """Process child result."""
        return ParentFinalEvent(final_result=ev.result + 10)


async def test_simple_nested_workflow() -> None:
    """Test basic nested workflow functionality."""
    workflow = SimpleParentWorkflow(timeout=5)
    result = await workflow.run()

    # Value should be: 5 -> doubled to 10 -> + 10 = 20
    assert result.final_result == 20


async def test_child_workflow_can_run_standalone() -> None:
    """Test that the child workflow can still run independently."""
    child = SimpleChildWorkflow(timeout=5)
    result = await child.run(value=7)

    assert result.result == 14


# ============================================================================
# Multi-step child workflow tests
# ============================================================================


class ProcessStartEvent(StartEvent):
    """Start event for processing workflow."""

    text: str


class IntermediateEvent(Event):
    """Intermediate event in child workflow."""

    text: str


class ProcessStopEvent(StopEvent):
    """Stop event for processing workflow."""

    processed: str


class TriggerProcessEvent(Event):
    """Trigger event for child workflow."""

    text: str


class ProcessCompleteEvent(Event):
    """Completion event from child workflow."""

    processed: str


class FinalStopEvent(StopEvent):
    """Final stop event."""

    output: str


class MultiStepChildWorkflow(Workflow):
    """A child workflow with multiple steps."""

    @step
    async def step1(self, ev: ProcessStartEvent) -> IntermediateEvent:
        """First step: uppercase."""
        return IntermediateEvent(text=ev.text.upper())

    @step
    async def step2(self, ev: IntermediateEvent) -> ProcessStopEvent:
        """Second step: add prefix."""
        return ProcessStopEvent(processed=f"PROCESSED: {ev.text}")


class MultiStepParentWorkflow(Workflow):
    """Parent workflow with multi-step child."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.include_workflow(
            MultiStepChildWorkflow,
            start_event=TriggerProcessEvent,
            stop_event=ProcessCompleteEvent,
        )

    @step
    async def start(self, ev: StartEvent) -> TriggerProcessEvent:
        """Start the workflow."""
        return TriggerProcessEvent(text="hello")

    @step
    async def finalize(self, ev: ProcessCompleteEvent) -> FinalStopEvent:
        """Finalize."""
        return FinalStopEvent(output=ev.processed)


async def test_multi_step_nested_workflow() -> None:
    """Test nested workflow with multiple steps."""
    workflow = MultiStepParentWorkflow(timeout=5)
    result = await workflow.run()

    assert result.output == "PROCESSED: HELLO"


# ============================================================================
# Context sharing tests
# ============================================================================


class CounterState(BaseModel):
    """State model for counter."""

    count: int = Field(default=0)


class CounterChildStart(StartEvent):
    """Start event for counter child."""

    increment: int


class CounterChildStop(StopEvent):
    """Stop event for counter child."""

    new_count: int


class TriggerCounterEvent(Event):
    """Trigger counter event."""

    increment: int


class CounterCompleteEvent(Event):
    """Counter complete event."""

    new_count: int


class CounterFinalEvent(StopEvent):
    """Final counter event."""

    total: int


class CounterChildWorkflow(Workflow):
    """Child workflow that increments a counter in shared state."""

    @step
    async def increment(
        self, ctx: Context[CounterState], ev: CounterChildStart
    ) -> CounterChildStop:
        """Increment the shared counter."""
        async with ctx.store.edit_state() as state:
            state.count += ev.increment

        state = await ctx.store.get_state()
        return CounterChildStop(new_count=state.count)


class CounterParentWorkflow(Workflow):
    """Parent workflow that shares state with child."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.include_workflow(
            CounterChildWorkflow,
            start_event=TriggerCounterEvent,
            stop_event=CounterCompleteEvent,
        )

    @step
    async def start(
        self, ctx: Context[CounterState], ev: StartEvent
    ) -> TriggerCounterEvent:
        """Initialize state and trigger child."""
        async with ctx.store.edit_state() as state:
            state.count = 10

        return TriggerCounterEvent(increment=5)

    @step
    async def finalize(
        self, ctx: Context[CounterState], ev: CounterCompleteEvent
    ) -> CounterFinalEvent:
        """Verify shared state was updated."""
        state = await ctx.store.get_state()
        # State should be updated by child
        assert state.count == ev.new_count
        return CounterFinalEvent(total=state.count)


async def test_context_sharing() -> None:
    """Test that parent and child workflows share the same context."""
    workflow = CounterParentWorkflow(timeout=5)
    ctx = Context(workflow)
    result = await workflow.run(ctx=ctx)

    # Counter started at 10, incremented by 5 = 15
    assert result.total == 15


# ============================================================================
# Edge case tests
# ============================================================================


async def test_multiple_workflow_instances() -> None:
    """Test creating multiple instances of a workflow with included workflows."""
    workflow1 = SimpleParentWorkflow(timeout=5)
    workflow2 = SimpleParentWorkflow(timeout=5)

    result1 = await workflow1.run()
    result2 = await workflow2.run()

    assert result1.final_result == 20
    assert result2.final_result == 20


async def test_nested_workflow_validation() -> None:
    """Test that validation works correctly with nested workflows."""
    workflow = SimpleParentWorkflow(timeout=5)

    # Should not raise any validation errors
    assert workflow.validate() is False


async def test_child_start_stop_events_not_in_parent_event_list() -> None:
    """Test that child's StartEvent/StopEvent don't appear in parent's events."""
    workflow = SimpleParentWorkflow(timeout=5)

    # Get all events from the parent workflow
    events = workflow.events

    # Child's StartEvent and StopEvent should not be in the parent's event list
    assert ChildStartEvent not in events
    assert ChildStopEvent not in events

    # But parent's events should be present
    assert ParentTriggerEvent in events
    assert ParentCompleteEvent in events


async def test_namespace_prevents_conflicts() -> None:
    """Test that namespace prevents step name conflicts."""

    class ConflictChildStart(StartEvent):
        value: int

    class ConflictChildStop(StopEvent):  # type: ignore
        final: int

    class TriggerConflict1(Event):
        value: int

    class CompleteConflict1(Event):
        result: int

    class TriggerConflict2(Event):
        value: int

    class CompleteConflict2(Event):
        result: int

    class ConflictChild(Workflow):
        @step
        async def process(self, ev: ConflictChildStart) -> ConflictChildStop:
            return ConflictChildStop(final=ev.value)

    class ConflictParentStop(StopEvent):
        total: int

    class ConflictParent(Workflow):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            # Include the same child workflow twice with different namespaces
            self.include_workflow(
                ConflictChild,
                start_event=TriggerConflict1,
                stop_event=CompleteConflict1,
                namespace="child1",
            )
            self.include_workflow(
                ConflictChild,
                start_event=TriggerConflict2,
                stop_event=CompleteConflict2,
                namespace="child2",
            )

        @step
        async def start(self, ev: StartEvent) -> TriggerConflict1:
            return TriggerConflict1(value=5)

        @step
        async def trigger_second(self, ev: CompleteConflict1) -> TriggerConflict2:
            return TriggerConflict2(value=ev.result * 2)

        @step
        async def finalize(self, ev: CompleteConflict2) -> ConflictParentStop:
            return ConflictParentStop(total=ev.result)

    workflow = ConflictParent(timeout=5)
    result = await workflow.run()

    # 5 -> 5 -> 10 -> 10
    assert result.total == 10


async def test_invalid_event_types_raise_error() -> None:
    """Test that non-Event types raise validation errors."""

    class InvalidParent(Workflow):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            with pytest.raises(WorkflowValidationError):
                self.include_workflow(
                    SimpleChildWorkflow,
                    # Invalid: not an Event subclass
                    start_event=str,  # type: ignore
                    stop_event=ParentCompleteEvent,
                )

        @step
        async def start(self, ev: StartEvent) -> StopEvent:
            return StopEvent()


# ============================================================================
# Complex nesting tests
# ============================================================================


class Level3Start(StartEvent):
    value: int


class Level3Stop(StopEvent):
    final: int


class TriggerLevel3(Event):
    value: int


class Level3Complete(Event):
    result: int


class Level2Start(StartEvent):
    value: int


class Level2Stop(StopEvent):
    final: int


class TriggerLevel2(Event):
    value: int


class Level2Complete(Event):
    result: int


class Level1Stop(StopEvent):
    final: int


class Level3Workflow(Workflow):
    """Innermost workflow."""

    @step
    async def process(self, ev: Level3Start) -> Level3Stop:
        return Level3Stop(final=ev.value + 1)


class Level2Workflow(Workflow):
    """Middle workflow that includes Level3."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.include_workflow(
            Level3Workflow,
            start_event=TriggerLevel3,
            stop_event=Level3Complete,
            namespace="level3",
        )

    @step
    async def start(self, ev: Level2Start) -> TriggerLevel3:
        return TriggerLevel3(value=ev.value + 10)

    @step
    async def finalize(self, ev: Level3Complete) -> Level2Stop:
        return Level2Stop(final=ev.result + 100)


class Level1Workflow(Workflow):
    """Outermost workflow that includes Level2."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.include_workflow(
            Level2Workflow,
            start_event=TriggerLevel2,
            stop_event=Level2Complete,
            namespace="level2",
        )

    @step
    async def start(self, ev: StartEvent) -> TriggerLevel2:
        return TriggerLevel2(value=1)

    @step
    async def finalize(self, ev: Level2Complete) -> Level1Stop:
        return Level1Stop(final=ev.result + 1000)


async def test_deeply_nested_workflows() -> None:
    """Test workflows nested multiple levels deep."""
    workflow = Level1Workflow(timeout=5)
    result = await workflow.run()

    # 1 -> (+10) 11 -> (+1) 12 -> (+100) 112 -> (+1000) 1112
    assert result.final == 1112
