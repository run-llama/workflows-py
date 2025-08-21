import pytest

from workflows import Workflow, Context, step
from typing import List
from workflows.events import (
    InternalDispatchEvent,
    _InProgressStepEvent,
    _QueueStateEvent,
    _RunningStepEvent,
    Event,
    StartEvent,
    StopEvent,
    EventType,
)


class SomeEvent(Event):
    data: str


class ExampleWorkflow(Workflow):
    @step
    async def first_step(self, ev: StartEvent, ctx: Context) -> SomeEvent:
        return SomeEvent(data=ev.message)

    @step
    async def second_step(self, ev: SomeEvent, ctx: Context) -> StopEvent:
        return StopEvent(result=ev.data)


@pytest.fixture()
def wf() -> ExampleWorkflow:
    return ExampleWorkflow()


@pytest.mark.asyncio
async def test_internal_events(wf: ExampleWorkflow) -> None:
    handler = wf.run()
    evs: List[EventType] = []
    async for ev in handler.stream_events(expose_internal=True):
        if isinstance(ev, InternalDispatchEvent):
            evs.append(type(ev.data))
    assert len(evs) > 0
    assert _InProgressStepEvent in evs
    assert _RunningStepEvent in evs
    assert _QueueStateEvent in evs
