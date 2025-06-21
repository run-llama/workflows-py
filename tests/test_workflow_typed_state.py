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
        await ctx.state.set("name", "John")
        await ctx.state.set("age", 30)

        # Get and update entire state
        state = await ctx.state.get_all()
        state.age += 1
        await ctx.state.set_all(state)

        return StopEvent()


@pytest.mark.asyncio
async def test_typed_state() -> None:
    wf = MyWorkflow()

    handler = wf.run()

    # Run the workflow
    _ = await handler

    # Check final state
    state = await handler.ctx.state.get_all()
    assert state.model_dump() == MyState(name="John", age=31).model_dump()
