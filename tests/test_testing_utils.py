import pytest

from workflows.events import (
    StartEvent,
    StopEvent,
    Event,
    StepStateChanged,
    EventsQueueChanged,
)
from workflows import Context, Workflow, step
from workflows.testing import WorkflowTestRunner
from workflows.testing.runner import WorkflowTestResult


class SecondEvent(Event):
    greeting: str


class SimpleWf(Workflow):
    @step
    async def step_one(self, ev: StartEvent, ctx: Context) -> SecondEvent:
        async with ctx.store.edit_state() as state:
            state.test = "this is a test"
        ctx.write_event_to_stream(SecondEvent(greeting="hello"))
        return SecondEvent(greeting="hello")

    @step
    async def step_two(self, ev: SecondEvent, ctx: Context) -> StopEvent:
        async with ctx.store.edit_state() as state:
            state.hello = "hello"
        return StopEvent(result="done")


@pytest.mark.asyncio
async def test_testing_utils() -> None:
    wf = SimpleWf()
    runner = WorkflowTestRunner(wf)
    wf_test_run = await runner.run(
        start_event=StartEvent(message="hi"),  # type: ignore
        exclude_events=[EventsQueueChanged],
    )
    assert isinstance(wf_test_run, WorkflowTestResult)
    assert len(wf_test_run.collected) == sum(
        [wf_test_run.event_types[k] for k in wf_test_run.event_types]
    )
    assert wf_test_run.event_types.get(EventsQueueChanged, 0) == 0
    assert wf_test_run.event_types.get(SecondEvent, 0) == 1
    assert wf_test_run.event_types.get(StopEvent, 0) == 1
    assert wf_test_run.event_types.get(StepStateChanged, 0) == len(
        wf_test_run.collected
    ) - (
        wf_test_run.event_types.get(SecondEvent, 0)
        + wf_test_run.event_types.get(StopEvent, 0)
    )
    assert str(wf_test_run.result) == "done"
