# LlamaIndex Workflows

LlamaIndex Workflows are a framework for orchestrating and chaining together complex systems of steps and events.

## What can you build with Workflows?

Workflows shine when you need to orchestrate complex, multi-step processes that involve AI models, APIs, and decision-making. Here are some examples of what you can build:

- **AI Agents** - Create intelligent systems that can reason, make decisions, and take actions across multiple steps
- **Document Processing Pipelines** - Build systems that ingest, analyze, summarize, and route documents through various processing stages
- **Multi-Model AI Applications** - Coordinate between different AI models (LLMs, vision models, etc.) to solve complex tasks
- **Research Assistants** - Develop workflows that can search, analyze, synthesize information, and provide comprehensive answers
- **Content Generation Systems** - Create pipelines that generate, review, edit, and publish content with human-in-the-loop approval
- **Customer Support Automation** - Build intelligent routing systems that can understand, categorize, and respond to customer inquiries

The async-first, event-driven architecture makes it easy to build workflows that can route between different capabilities, implement parallel processing patterns, loop over complex sequences, and maintain state across multiple steps - all the features you need to make your AI applications production-ready.

## Key Features

- **async-first** - workflows are built around python's async functionality - steps are async functions that process incoming events from an asyncio queue and emit new events to other queues. This also means that workflows work best in your async apps like FastAPI, Jupyter Notebooks, etc.
- **event-driven** - workflows consist of steps and events. Organizing your code around events and steps makes it easier to reason about and test.
- **state management** - each run of a workflow is self-contained, meaning you can launch a workflow, save information within it, serialize the state of a workflow and resume it later.
- **observability** - workflows are automatically instrumented for observability, meaning you can use tools like `Arize Phoenix` and `OpenTelemetry` right out of the box.

## Quick Start

Install the package:

```bash
pip install llama-index-workflows
```

And create your first workflow:

```python
import asyncio
from pydantic import BaseModel, Field
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent

class MyEvent(Event):
    msg: list[str]

class RunState(BaseModel):
    num_runs: int = Field(default=0)

class MyWorkflow(Workflow):
    @step
    async def start(self, ctx: Context[RunState], ev: StartEvent) -> MyEvent:
        async with ctx.store.edit_state() as state:
            state.num_runs += 1

            return MyEvent(msg=[ev.input_msg] * state.num_runs)

    @step
    async def process(self, ctx: Context[RunState], ev: MyEvent) -> StopEvent:
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

Visit the [complete documentation](https://docs.llamaindex.ai/en/stable/understanding/workflows/) for more examples using `llama-index`!
