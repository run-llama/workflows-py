# LlamaIndex Workflows

LlamaIndex Workflows are a framework for orchestrating and chaining together complex systems of steps and events.

Using Workflows, you get access to an async-first event-driven system for building and running your own workflows.

## Key Features

- **async-first** - workflows are built around python's async functionality - steps are async functions that process incoming events from an asyncio queue and emit new events to other queues. This also means that workflows work best in your async apps like FastAPI, Jupyter Notebooks, etc.
- **event-driven** - workflows consist of steps and events. Organizing your code around events and steps makes it easier to reason about and test.
- **state management** - each run of a workflow is self-contained, meaning you can launch a workflow, save information within it, serialize the state of a workflow and resume it later.
- **observability** - workflows are automatically instrumented for observability, meaning you can use tools like [Arize Pheonix] and [OpenTelemetry] right out of the box.

## Quick Start

Install the package:

```bash
pip install llama-index-workflows
```

And create your first workflow:

```python
import asyncio
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent

class MyEvent(Event):
    msg: list[str]

class MyWorkflow(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> MyEvent:
        num_runs = await ctx.get("num_runs", default=0)
        num_runs += 1
        await ctx.set("num_runs", num_runs)

        return MyEvent(msg=[ev.input_msg] * num_runs)
    
    @step
    async def process(self, ctx: Context, ev: MyEvent) -> StopEvent:
        data_length = len("".join(ev.msg))
        new_msg = f"Processed {len(ev.msg)} times, data length: {data_length}"
        return StopEvent(result=new_msg)

async def main():
    workflow = MyWorkflow()

    # [optional] provide a context object to the workflow
    ctx = Context(workflow)
    result = await workflow.run(input_msg="Hello, world!", ctx=ctx)
    print("Workflow result:", result)

    # re-running with the same context will retain the state
    result = await workflow.run(input_msg="Hello, world!", ctx=ctx)
    print("Workflow result:", result)

if __name__ == "__main__":
    asyncio.run(main())
```

In the example above
- Steps that accept a `StartEvent` will be run first.
- Steps that return a `StopEvent` will end the workflow.
- Intermediate events are user defined and can be used to pass information between steps.
- The `Context` object is also used to share information between steps.

Visit the [complete documentation](https://docs.llamaindex.ai/en/stable/understanding/workflows/) for more examples!
