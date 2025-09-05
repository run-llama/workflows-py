import pytest
import asyncio

from workflows import Workflow, Context, step
from typing import List, Union
from workflows.events import (
    InternalDispatchEvent,
    EventsQueueChanged,
    StepStateChanged,
    StepState,
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


class ExampleWorkflowDictState(Workflow):
    @step
    async def first_step(self, ev: StartEvent, ctx: Context) -> SomeEvent:
        async with ctx.store.edit_state() as state:
            state.test = "Test"
        return SomeEvent(data=ev.message)

    @step
    async def second_step(self, ev: SomeEvent, ctx: Context) -> StopEvent:
        async with ctx.store.edit_state() as state:
            del state._data["test"]
        return StopEvent(result=ev.data)


class ExampleWorkflowMultiWorkers(Workflow):
    @step
    async def first_step(self, ev: StartEvent, ctx: Context) -> Union[SomeEvent, None]:
        for _ in range(10):
            ctx.send_event(SomeEvent(data=ev.message))
        return None

    @step(num_workers=10)
    async def second_step(self, ev: SomeEvent, ctx: Context) -> StopEvent:
        # allow for each worker to process each event
        await asyncio.sleep(0.1)
        return StopEvent(result=ev.data)


@pytest.fixture()
def wf() -> ExampleWorkflow:
    return ExampleWorkflow()


@pytest.fixture()
def wf_state() -> ExampleWorkflowState:
    return ExampleWorkflowState()


@pytest.fixture()
def wf_workers() -> ExampleWorkflowMultiWorkers:
    return ExampleWorkflowMultiWorkers()


@pytest.fixture()
def wf_dict_state() -> ExampleWorkflowDictState:
    return ExampleWorkflowDictState()


@pytest.mark.asyncio
async def test_internal_events(wf: ExampleWorkflow) -> None:
    handler = wf.run(message="hello")
    evs: List[EventType] = []
    async for ev in handler.stream_events(expose_internal=True):
        if isinstance(ev, InternalDispatchEvent):
            evs.append(type(ev))
    await handler
    assert len(evs) > 0
    assert all(ev == StepStateChanged or ev == EventsQueueChanged for ev in evs)


@pytest.mark.asyncio
async def test_internal_events_state(wf_state: ExampleWorkflowState) -> None:
    handler = wf_state.run(message="hello")
    evs: List[StepStateChanged] = []
    async for ev in handler.stream_events(expose_internal=True):
        if (
            isinstance(ev, StepStateChanged)
            and ev.name != "_done"
            and ev.context_state is not None
        ):
            evs.append(ev)
    await handler
    assert len(evs) == 4
    for i, ev in enumerate(evs):
        if i < 2:
            assert ev.name == "first_step"
        else:
            assert ev.name == "second_step"
        if i % 2 == 0:
            assert ev.step_state == StepState.PREPARING
        else:
            assert ev.step_state == StepState.NOT_IN_PROGRESS
    assert evs[0].context_state["state_data"] == {"test": '""'}  # type: ignore
    assert all(ev.context_state["state_data"] == {"test": '"Test"'} for ev in evs[1:])  # type: ignore


@pytest.mark.asyncio
async def test_internal_events_dict_state(
    wf_dict_state: ExampleWorkflowDictState,
) -> None:
    # prove that state modification works also with DictState
    handler = wf_dict_state.run(message="hello")
    evs: List[StepStateChanged] = []
    async for ev in handler.stream_events(expose_internal=True):
        if (
            isinstance(ev, StepStateChanged)
            and ev.name != "_done"
            and ev.context_state is not None
        ):
            evs.append(ev)
    await handler
    assert len(evs) == 4
    for i, ev in enumerate(evs):
        if i < 2:
            assert ev.name == "first_step"
        else:
            assert ev.name == "second_step"
        if i % 2 == 0:
            assert ev.step_state == StepState.PREPARING
        else:
            assert ev.step_state == StepState.NOT_IN_PROGRESS
    assert evs[0].context_state["state_data"] == {}  # type: ignore
    assert evs[1].context_state["state_data"] == {"test": '"Test"'}  # type: ignore
    assert evs[2].context_state["state_data"] == {"test": '"Test"'}  # type: ignore
    assert evs[3].context_state["state_data"] == {}  # type: ignore


@pytest.mark.asyncio
async def test_internal_events_multiple_workers(
    wf_workers: ExampleWorkflowMultiWorkers,
) -> None:
    handler = wf_workers.run(message="hello")
    run_ids = []
    async for ev in handler.stream_events(expose_internal=True):
        if isinstance(ev, StepStateChanged):
            # avoid duplication of the run_ids and exclude the "_done" step
            if ev.step_state == StepState.RUNNING and ev.name != "_done":
                run_ids.append(ev.worker_id)
    await handler
    assert len(run_ids) == 11
    assert (
        len(set(run_ids)) == 11
    )  # this check proves that we can differentiate among the same steps when they are run by different workers
