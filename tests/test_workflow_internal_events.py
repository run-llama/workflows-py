import pytest
import asyncio

from workflows import Workflow, Context, step
from workflows.testing import WorkflowTestRunner
from typing import Union
from workflows.events import (
    EventsQueueChanged,
    StepStateChanged,
    StepState,
    Event,
    StartEvent,
    StopEvent,
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
    test_runner = WorkflowTestRunner(wf)
    result = await test_runner.run(
        start_event=StartEvent(message="hello"),  # type: ignore
        exclude_events=[StopEvent],
    )
    assert len(result.collected) > 0
    assert all(
        isinstance(ev, StepStateChanged) or isinstance(ev, EventsQueueChanged)
        for ev in result.collected
    )


@pytest.mark.asyncio
async def test_internal_events_state(wf_state: ExampleWorkflowState) -> None:
    test_runner = WorkflowTestRunner(wf_state)
    result = await test_runner.run(
        start_event=StartEvent(message="hello"),  # type: ignore
        exclude_events=[StopEvent, EventsQueueChanged],
    )
    assert all(isinstance(ev, StepStateChanged) for ev in result.collected)
    filtered_events = [
        r for r in result.collected if r.context_state is not None and r.name != "_done"
    ]
    assert len(filtered_events) == 4
    for i, ev in enumerate(filtered_events):
        if i < 2:
            assert ev.name == "first_step"
        else:
            assert ev.name == "second_step"
        if i % 2 == 0:
            assert ev.step_state == StepState.PREPARING
        else:
            assert ev.step_state == StepState.NOT_IN_PROGRESS
    assert filtered_events[0].context_state["state_data"] == {"test": '""'}
    assert all(
        ev.context_state["state_data"] == {"test": '"Test"'}
        for ev in filtered_events[1:]
    )


@pytest.mark.asyncio
async def test_internal_events_dict_state(
    wf_dict_state: ExampleWorkflowDictState,
) -> None:
    # prove that state modification works also with DictState
    test_runner = WorkflowTestRunner(wf_dict_state)
    result = await test_runner.run(
        start_event=StartEvent(message="hello"),  # type: ignore
        exclude_events=[StopEvent, EventsQueueChanged],
    )
    assert all(isinstance(ev, StepStateChanged) for ev in result.collected)
    filtered_events = [
        r for r in result.collected if r.context_state is not None and r.name != "_done"
    ]
    assert len(filtered_events) == 4
    for i, ev in enumerate(filtered_events):
        if i < 2:
            assert ev.name == "first_step"
        else:
            assert ev.name == "second_step"
        if i % 2 == 0:
            assert ev.step_state == StepState.PREPARING
        else:
            assert ev.step_state == StepState.NOT_IN_PROGRESS
    assert filtered_events[0].context_state["state_data"] == {}
    assert filtered_events[1].context_state["state_data"] == {"test": '"Test"'}
    assert filtered_events[2].context_state["state_data"] == {"test": '"Test"'}
    assert filtered_events[3].context_state["state_data"] == {}


@pytest.mark.asyncio
async def test_internal_events_multiple_workers(
    wf_workers: ExampleWorkflowMultiWorkers,
) -> None:
    test_runner = WorkflowTestRunner(wf_workers)
    result = await test_runner.run(
        start_event=StartEvent(message="hello"),  # type: ignore
        exclude_events=[StopEvent, EventsQueueChanged],
    )
    assert all(isinstance(ev, StepStateChanged) for ev in result.collected)
    run_ids = [
        r.worker_id
        for r in result.collected
        if r.step_state == StepState.RUNNING and r.name != "_done"
    ]
    assert len(run_ids) == 11
    assert (
        len(set(run_ids)) == 11
    )  # this check proves that we can differentiate among the same steps when they are run by different workers
