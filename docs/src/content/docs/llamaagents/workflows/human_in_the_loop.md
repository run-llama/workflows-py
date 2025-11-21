---
sidebar:
  order: 6
title: Human in the Loop
---

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

If needed, you can also subclass these two events to add custom payloads.

## Stopping/Resuming Between Human Responses

Also note that you can break out of the loop, and resume it later. This is useful if you want to pause the workflow to wait for a human response, but continue the workflow later.

```python
handler = workflow.run()
async for event in handler.stream_events():
    if isinstance(event, InputRequiredEvent):
        # Serialize the context, store it anywhere as a JSON blob
        ctx_dict = handler.ctx.to_dict()
        await handler.cancel_run()
        break

...

# now we handle the human response once it comes in
response = input(event.prefix)

restored_ctx = Context.from_dict(workflow, ctx_dict)
handler = workflow.run(ctx=restored_ctx)

# Send the event to resume the workflow
handler.ctx.send_event(HumanResponseEvent(response=response))

# now we resume the workflow streaming with our restored context
async for event in handler.stream_events():
    continue

final_result = await handler
```
