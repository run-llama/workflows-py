import asyncio
import pytest
from pydantic import BaseModel, Field
from typing import Union, Optional

from workflows import Context, Workflow
from workflows.testing import WorkflowTestRunner
from workflows.decorators import step
from workflows.events import StartEvent, StopEvent, Event


class MyState(BaseModel):
    name: str = Field(default="Jane")
    age: int = Field(default=25)


class MyWorkflow(Workflow):
    @step
    async def step(self, ctx: Context[MyState], ev: StartEvent) -> StopEvent:
        # Modify state attributes
        await ctx.store.set("name", "John")
        await ctx.store.set("age", 30)

        # Get and update entire state
        state = await ctx.store.get_state()
        state.age += 1
        await ctx.store.set_state(state)

        return StopEvent()


@pytest.mark.asyncio
async def test_typed_state() -> None:
    test_runner = WorkflowTestRunner(MyWorkflow())

    await test_runner.run()

    # Check final state
    ctx = test_runner._workflow._contexts.pop()
    assert ctx is not None
    state = await ctx.store.get_state()
    assert state.model_dump() == MyState(name="John", age=31).model_dump()


class SomeState(BaseModel):
    val: int = Field(default=0)


class WorkerEvent(Event):
    pass


class ResultEvent(Event):
    pass


class GatherEvent(Event):
    pass


class ParallelWorkflow(Workflow):
    @step
    async def init(
        self, ctx: Context[SomeState], ev: StartEvent
    ) -> Union[WorkerEvent, GatherEvent]:
        for _ in range(10):
            ctx.send_event(WorkerEvent())

        return GatherEvent()

    @step
    async def worker(self, ctx: Context[SomeState], ev: WorkerEvent) -> ResultEvent:
        async with ctx.store.edit_state() as state:
            state.val += 1
            await asyncio.sleep(0.01)
            if state.val % 2 == 0:
                state.val -= 1

        return ResultEvent()

    @step
    async def gather(
        self, ctx: Context[SomeState], ev: Union[GatherEvent, ResultEvent]
    ) -> Optional[StopEvent]:
        results = ctx.collect_events(ev, [ResultEvent] * 10)
        if not results:
            return None

        state = await ctx.store.get_state()
        return StopEvent(result=state.val)


@pytest.mark.asyncio
async def test_typed_state_with_context_manager() -> None:
    test_runner = WorkflowTestRunner(ParallelWorkflow())

    result = await test_runner.run()

    # Should only be 1 since the context manager locks the state
    assert result.result == 1
