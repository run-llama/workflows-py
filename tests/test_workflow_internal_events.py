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
async def test_internal_events_sequence(wf_state: ExampleWorkflowState) -> None:
    test_runner = WorkflowTestRunner(wf_state)
    result = await test_runner.run(
        start_event=StartEvent(message="hello"),  # type: ignore
        exclude_events=[StopEvent, EventsQueueChanged],
    )
    assert all(isinstance(ev, StepStateChanged) for ev in result.collected)
    filtered_events = [
        {"name": x.name, "step_state": x.step_state} for x in result.collected
    ]
    assert filtered_events == [
        dict(name="first_step", step_state=StepState.PREPARING),
        dict(name="first_step", step_state=StepState.IN_PROGRESS),
        dict(name="first_step", step_state=StepState.RUNNING),
        dict(name="first_step", step_state=StepState.NOT_RUNNING),
        dict(name="first_step", step_state=StepState.NOT_IN_PROGRESS),
        # TODO(adrian) very soon! - duplicate event bug. Encountered while refactoring this test.
        dict(name="first_step", step_state=StepState.NOT_IN_PROGRESS),
        dict(name="first_step", step_state=StepState.EXITED),
        dict(name="second_step", step_state=StepState.PREPARING),
        dict(name="second_step", step_state=StepState.IN_PROGRESS),
        dict(name="second_step", step_state=StepState.RUNNING),
        dict(name="second_step", step_state=StepState.NOT_RUNNING),
        dict(name="second_step", step_state=StepState.NOT_IN_PROGRESS),
        # TODO(adrian) very soon! - duplicate event bug. Encountered while refactoring this test.
        dict(name="second_step", step_state=StepState.NOT_IN_PROGRESS),
        dict(name="second_step", step_state=StepState.EXITED),
        dict(name="_done", step_state=StepState.PREPARING),
        dict(name="_done", step_state=StepState.IN_PROGRESS),
        dict(name="_done", step_state=StepState.RUNNING),
    ]


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
