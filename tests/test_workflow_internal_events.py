import pytest

from workflows import Workflow, Context, step
from typing import List
from workflows.events import (
    InternalDispatchEvent,
    _InProgressStepEvent,
    _QueueStateEvent,
    _RunningStepEvent,
    _StateModificationEvent,
    _StepEmitEvent,
    Event,
    StartEvent,
    StopEvent,
    EventType,
)
from pydantic import BaseModel


class SomeEvent(Event):
    data: str


class ExampleWorkflow(Workflow):
    @step
    async def first_step(self, ev: StartEvent, ctx: Context) -> SomeEvent:
        return SomeEvent(data=ev.message)

    @step
    async def second_step(self, ev: SomeEvent, ctx: Context) -> StopEvent:
        return StopEvent(result=ev.data)


class WfState(BaseModel):
    test: str = ""


class ExampleWorkflowState(Workflow):
    @step
    async def first_step(self, ev: StartEvent, ctx: Context[WfState]) -> SomeEvent:
        async with ctx.store.edit_state() as state:
            state.test = "Test"
        return SomeEvent(data=ev.message)

    @step
    async def second_step(self, ev: SomeEvent, ctx: Context[WfState]) -> StopEvent:
        return StopEvent(result=ev.data)


@pytest.fixture()
def wf() -> ExampleWorkflow:
    return ExampleWorkflow()


@pytest.fixture()
def wf_state() -> ExampleWorkflowState:
    return ExampleWorkflowState()


@pytest.mark.asyncio
async def test_internal_events(wf: ExampleWorkflow) -> None:
    handler = wf.run(message="hello")
    evs: List[EventType] = []
    async for ev in handler.stream_events(expose_internal=True):
        if isinstance(ev, InternalDispatchEvent):
            evs.append(type(ev))
    await handler
    assert len(evs) > 0
    assert _InProgressStepEvent in evs
    assert _RunningStepEvent in evs
    assert _QueueStateEvent in evs
    assert _StepEmitEvent in evs
    assert evs.count(_StepEmitEvent) == 2


@pytest.mark.asyncio
async def test_internal_events_state(wf_state: ExampleWorkflowState) -> None:
    handler = wf_state.run(message="hello")
    evs: List[EventType] = []
    async for ev in handler.stream_events(expose_internal=True):
        if isinstance(ev, InternalDispatchEvent):
            print(f"Event {type(ev)}: {ev}")
            evs.append(type(ev))
    await handler
    assert len(evs) > 0
    assert _StateModificationEvent in evs
    assert evs.count(_StateModificationEvent) == 1
