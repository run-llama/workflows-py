# Managing events

## Waiting for Multiple Events

The context does more than just hold data, it also provides utilities to buffer and wait for multiple events.

For example, you might have a step that waits for a query and retrieved nodes before synthesizing a response:

```python
from llama_index.core import get_response_synthesizer


@step
async def synthesize(
    self, ctx: Context, ev: QueryEvent | RetrieveEvent
) -> StopEvent | None:
    data = ctx.collect_events(ev, [QueryEvent, RetrieveEvent])
    # check if we can run
    if data is None:
        return None

    # unpack -- data is returned in order
    query_event, retrieve_event = data

    # run response synthesis
    synthesizer = get_response_synthesizer()
    response = synthesizer.synthesize(
        query_event.query, nodes=retrieve_event.nodes
    )

    return StopEvent(result=response)
```

Using `ctx.collect_events()` we can buffer and wait for ALL expected events to arrive. This function will only return data (in the requested order) once all events have arrived.

## Manually Triggering Events

Normally, events are triggered by returning another event during a step. However, events can also be manually dispatched using the `ctx.send_event(event)` method within a workflow.

Here is a short toy example showing how this would be used:

```python
from workflows import Workflow, step
from workflows.events import (
    Event,
    StartEvent,
    StopEvent,
)


class MyEvent(Event):
    pass


class MyEventResult(Event):
    result: str


class GatherEvent(Event):
    pass


class MyWorkflow(Workflow):
    @step
    async def dispatch_step(
        self, ctx: Context, ev: StartEvent
    ) -> MyEvent | GatherEvent:
        ctx.send_event(MyEvent())
        ctx.send_event(MyEvent())

        return GatherEvent()

    @step
    async def handle_my_event(self, ev: MyEvent) -> MyEventResult:
        return MyEventResult(result="result")

    @step
    async def gather(
        self, ctx: Context, ev: GatherEvent | MyEventResult
    ) -> StopEvent | None:
        # wait for events to finish
        events = ctx.collect_events(ev, [MyEventResult, MyEventResult])
        if not events:
            return None

        return StopEvent(result=events)
```

## Streaming Events

You can also iterate over events as they come in. This is useful for streaming purposes, showing progress, or for debugging. The handler object will emit events that are explicitly written to the stream using `ctx.write_event_to_stream()`:

```python
class ProgressEvent(Event):
    msg: str


class MyWorkflow(Workflow):
    @step
    async def step_one(self, ctx: Context, ev: StartEvent) -> FirstEvent:
        ctx.write_event_to_stream(ProgressEvent(msg="Step one is happening"))
        return FirstEvent(first_output="First step complete.")
```

You can then pick up the events like this:

```python
w = MyWorkflow(...)

handler = w.run(topic="Pirates")

async for event in handler.stream_events():
    print(event)

result = await handler
```

There is also the possibility of streaming internal events (such as changes occurring while the step is running, modifications of the workflow state or variations in the size of the internal queues). In order to do so, you need to pass `expose_internal = True` to `stream_events`.

All the internal events are instances of `InternalDispatchEvent`, but they can be divided into two sub-classes:

- `StepStateChanged`: exposes internal changes in the state of the event, including whether the step is running or in progress, what worker it is running on and what events it takes as input and output, as well as changes in the workflow state.
- `EventsQueueChanged`: reports the state of the internal queues.

Here is how you can stream these internal events:

```python
w = MyWorkflow(...)

handler = w.run(topic="Pirates")

async for event in handler.stream_events(expose_internal=True):
    if isinstance(event, StepStateChanged):
        print("Name of current step:", event.name)
        print("State of current step:", event.step_state.value)
        print("Input event for current step:", event.input_event_name)
        print("Workflow state at current step:", event.context_state or "No state reported")
        print("Output event of current step:", event.output_event_name or "No output event yet")
    elif isinstance(event, EventsQueueChanged):
        print("Queue name:", event.name)
        print("Queue size:", event.size)
    # other event streaming logic here

result = await handler
```

## Human-in-the-loop

Since workflows are so flexible, there are many possible ways to implement human-in-the-loop patterns.

The easiest way to implement a human-in-the-loop is to use the `InputRequiredEvent` and `HumanResponseEvent` events during event streaming.

```python
from workflows.events import InputRequiredEvent, HumanResponseEvent


class HumanInTheLoopWorkflow(Workflow):
    @step
    async def step1(self, ev: StartEvent) -> InputRequiredEvent:
        return InputRequiredEvent(prefix="Enter a number: ")

    @step
    async def step2(self, ev: HumanResponseEvent) -> StopEvent:
        return StopEvent(result=ev.response)


# workflow should work with streaming
workflow = HumanInTheLoopWorkflow()

handler = workflow.run()
async for event in handler.stream_events():
    if isinstance(event, InputRequiredEvent):
        # here, we can handle human input however you want
        # this means using input(), websockets, accessing async state, etc.
        # here, we just use input()
        response = input(event.prefix)
        handler.ctx.send_event(HumanResponseEvent(response=response))

final_result = await handler
```

Here, the workflow will wait until the `HumanResponseEvent` is emitted.

Also note that you can break out of the loop, and resume it later. This is useful if you want to pause the workflow to wait for a human response, but continue the workflow later.

```python
handler = workflow.run()
async for event in handler.stream_events():
    if isinstance(event, InputRequiredEvent):
        break

# now we handle the human response
response = input(event.prefix)
handler.ctx.send_event(HumanResponseEvent(response=response))

# now we resume the workflow streaming
async for event in handler.stream_events():
    continue

final_result = await handler
```
