import pytest
from pydantic import BaseModel, Field

from workflows import Context, Workflow
from workflows.decorators import step
from workflows.events import StartEvent, StopEvent


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
    wf = MyWorkflow()

    handler = wf.run()

    # Run the workflow
    _ = await handler

    # Check final state
    state = await handler.ctx.store.get_state()
    assert state.model_dump() == MyState(name="John", age=31).model_dump()
