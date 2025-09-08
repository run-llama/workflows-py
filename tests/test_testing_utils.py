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
    async with wf.run_test() as test_runner:
        assert isinstance(test_runner, WorkflowTestRunner)
        collected, freqs, res = await test_runner.run_and_collect(
            start_event=StartEvent(message="hi"),  # type: ignore
            exclude_events=[EventsQueueChanged],
        )
        assert len(collected) == sum([freqs[k] for k in freqs])
        assert freqs.get(EventsQueueChanged, 0) == 0
        assert freqs.get(SecondEvent, 0) == 1
        assert freqs.get(StopEvent, 0) == 1
        assert freqs.get(StepStateChanged, 0) == len(collected) - (
            freqs.get(SecondEvent, 0) + freqs.get(StopEvent, 0)
        )
        assert str(res) == "done"
        output_event = await test_runner.send_test_event(
            step="step_one",
            event=StartEvent(message="hello"),  # type: ignore
            ctx=Context(wf),
        )
        assert isinstance(output_event, SecondEvent)
        assert output_event.greeting == "hello"
