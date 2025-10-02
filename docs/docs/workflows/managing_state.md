# Managing State

By default, workflows automatically initialize and untyped state store. You can access this as needed to share information between workflow steps through the `Context` object.

```python
from workflows import Workflow, Context, step
from workflows.events import StartEvent, StopEvent

class MyWorkflow(Workflow):

    @step
    async def my_step(self, ctx: Context, ev: StartEvent) -> StopEvent:
       current_count = await ctx.store.get("count", default=0)
       current_count += 1
       await ctx.store.set("count", current_count)
       return StopEvent()
```

## Locking the State

There are cases where the state might be manipulated by multiple steps running at the same time. In these cases, in can be useful **lock** the state to prevent race conditions. You can do this by using the `Context` object's `edit_state` method:

```python
@step
async def my_step(self, ctx: Context, ev: StartEvent) -> StopEvent:
   # No other steps can access the state while the `with` block is running
   async with ctx.store.edit_state() as ctx_state:
       if "count" not in state:
           ctx_state["count"] = 0
       ctx_state["count"] += 1
   return StopEvent()
```

## Adding Typed State

Often, you'll have some pre-set shape that you want to use as the state for your workflow. The best way to do this is to use a `Pydantic` model to define the state. This way, you:

- Get type hints for your state
- Get automatic validation of your state
- (Optionally) Have full control over the serialization and deserialization of your state using [validators](https://docs.pydantic.dev/latest/concepts/validators/) and [serializers](https://docs.pydantic.dev/latest/concepts/serialization/#custom-serializers)

**NOTE:** You should use a pydantic model that has defaults for all fields. This enables the `Context` object to automatically initialize the state with the defaults.

Here's a quick example of how you can leverage workflows + pydantic to take advantage of all these features:

```python
from pydantic import BaseModel, Field


class CounterState(BaseModel):
    count: int = Field(default=0)
```

Then, simply annotate your workflow state with the state model:

```python
from workflows import Workflow, step
from workflows.events import (
    StartEvent,
    StopEvent,
)


class MyWorkflow(Workflow):
    @step
    async def start(
        self,
        ctx: Context[CounterState],
        ev: StartEvent
    ) -> StopEvent:
        # Allows for atomic state updates
        async with ctx.store.edit_state() as ctx_state:
            ctx_state.count += 1

        return StopEvent(result="Done!")
```

## Maintaining Context Across Runs

As you have seen, workflows have a `Context` object that can be used to maintain state across steps.

If you want to maintain state across multiple runs of a workflow, you can pass a previous context into the `.run()` method.

```python
workflow = MyWorkflow()
ctx = Context(workflow)

handler = workflow.run(ctx=ctx)
result = await handler

# Optional: save the ctx somewhere and restore
# ctx_dict = ctx.to_dict()
# ctx = Context.from_dict(workflow, ctx_dict)

# continue with next run
handler = w.run(ctx=ctx)
result = await handler
```
