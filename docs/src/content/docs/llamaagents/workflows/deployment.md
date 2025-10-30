---
title: Run Your Workflow as a Server
---

The `workflows` library includes a `WorkflowServer` class that allows you to easily expose your workflows over an HTTP API. This provides a flexible way to run and manage workflows from any HTTP-capable client.

Additionally, the `WorkflowServer` is deployed with a static debugging application that allows you to visualize, run, and debug workflows. This is automatically mounted at the root `/` path of the running server.

## Programmatic Usage

You can create a server, add your workflows, and run it programmatically. This is useful when you want to embed the
workflow server in a larger application.

First, create a Python file (e.g., `my_server.py`):

```python
# my_server.py
from workflows import Workflow, step
from workflows.context import Context
from workflows.events import Event, StartEvent, StopEvent
from workflows.server import WorkflowServer


class StreamEvent(Event):
    sequence: int


# Define a simple workflow
class GreetingWorkflow(Workflow):
    @step
    async def greet(self, ctx: Context, ev: StartEvent) -> StopEvent:
        for i in range(3):
            ctx.write_event_to_stream(StreamEvent(sequence=i))
            await asyncio.sleep(0.3)

        name = getattr(ev, "name", "World")
        return StopEvent(result=f"Hello, {name}!")
greet_wf = GreetingWorkflow()


# Create a server instance
server = WorkflowServer()

# Add the workflow to the server
server.add_workflow("greet", greet_wf)

# To run the server programmatically (e.g., from your own script)
# import asyncio
#
# async def main():
#     await server.serve(host="0.0.0.0", port=8080)
#
# if __name__ == "__main__":
#     asyncio.run(main())
```

## Command-Line Interface (CLI)

The library also provides a convenient CLI to run a server from a file containing a `WorkflowServer` instance.

Given the `my_server.py` file from the example above, you can start the server with the following command:

```bash
python -m workflows.server my_server.py
```

The server will start on `0.0.0.0:8080` by default. You can configure the host and port using the
`WORKFLOWS_PY_SERVER_HOST` and `WORKFLOWS_PY_SERVER_PORT` environment variables.

## Workflow Debugger UI

The `WorkflowServer` is deployed with a static debugging application that allows you to visualize, run, and debug workflows. This is automatically mounted at the root `/` path of the running server.

![Workflow Debugger UI](./assets/ui_sample.png)

The Workflow Debugging UI offers a few key features:
- **Workflow Visualization**: The UI provides a visual representation of the workflow's structure both statically and while it is running. You can re-arrange the nodes as needed.
- **Automatic schema detection**: If you customize the schemas of your start/stop events, or internal events, the UI will automatically detect and display UI appropriate for the schema.
- **Human-in-the-loop**: While a workflow is running, you can send any event into the workflow. This is useful for workflows that rely on human input to continue execution. See the `Send Event` button on the top of the events log.
- **Events Log**: All streamed events are logged in the UI, allowing you to inspect the workflow's execution in real-time in the right side-panel.
- **Multiple Runs**: Debug and compare multiple runs. Each time you run a workflow, the left-side panel tracks that run.
- **Multiple Workflows**: The UI will let you run any workflow that is mounted within the `WorkflowServer`.

### Handling "Hidden" Events

Sometimes, workflows will send/accept events that are annotated in the workflow (like using `ctx.wait_for_event()`). In these cases, you can still inform the UI about these events using the `server.add_workflow(..., additional_events=[...])` API to inject those events. Then, UI elements like the `Send Event` functionality will be aware of these events.

## API Endpoints

The `WorkflowServer` exposes the following RESTful endpoints:

| Method | Path                           | Description                                                                                             |
|--------|--------------------------------|---------------------------------------------------------------------------------------------------------|
| `GET`  | `/health`                      | Returns a health check response (`{"status": "healthy"}`).                                               |
| `GET`  | `/workflows`                   | Lists the names of all registered workflows.                                                            |
| `POST` | `/workflows/{name}/run`        | Runs the specified workflow synchronously and returns the final result.                                 |
| `POST` | `/workflows/{name}/run-nowait` | Starts the specified workflow asynchronously and returns a `handler_id`.                                |
| `GET`  | `/handlers/{handler_id}`        | Retrieves the result of an asynchronously run workflow. Returns `202 Accepted` if still running, `500` if the workflow failed, `200` if the workflow completed.       |
| `GET`  | `/events/{handler_id}`         | Streams all events from a running workflow as newline-delimited JSON (`application/x-ndjson` and `text/event-stream` if SSE are enabled).          |
| `POST`  | `/events/{handler_id}`         | Sends an event to a workflow during its execution (useful for human-in-the-loop)         |
| `GET`  | `/handlers`         |  Get all the workflow handlers (running and completed)        |
| `POST`  | `/handlers/{handler_id}/cancel`         | Stop and cancel the execution of a workflow.        |


### Running a Workflow (`/run`)

To run a workflow and wait for its completion, send a `POST` request to `/workflows/{name}/run`.

**Request Body:**

```json
{
  "start_event": {},
  "context": {},
  "handler_id": "",
}
```

- `start_event`: serialized representation of a StartEvent or a subclass of it. Using this as a workflow input is recommended.
- `context`: serialized representation of the workflow context
- `handler_id`: workflow handler identifier to continue from a previous completed run.

**Successful Response (`200 OK`):**

```json
{
  "result": "The workflow has been successfully run"
}
```

### Running a Workflow Asynchronously (`/run-nowait`)

To start a workflow without waiting for it to finish, use the `/run-nowait` endpoint.

**Request Body:**

```json
{
  "start_event": {},
  "context": {},
  "handler_id": ""
}
```

The request body has the same arguments as the `/run` endpoint.

**Successful Response (`200 OK`):**

```json
{
  "handler_id": "someUniqueId123",
  "status": "started"
}
```

You can then use the `handler_id` to check for the result or stream events.

## Streaming events (`GET /events/{handler_id}`)

> _This endpoint only works if you previously started a workflow asynchronously with `/run-nowait`_

To stream events either as Server-Sent Events (SSE) or as multi-line JSON payloads, you can send a request to the `/events/{handler_id}` endpoint with the handler ID of an asynchronous workflow run you previously started.

**Query parameters**

- `sse` (set to either "true" or "false", not required): stream the events as Server Sent Events (`text/event-stream`) if true, else stream them as a multi-line JSON payload (`application/x-ndjson`). Defaults to true.
- `acquire_timeout` (a float-convertible string, not required): timeout for acquiring the lock to iterate over events
- `include_internal` (set to either "true" or "false", not required): stream internal workfloe events if set to true. Defaults to false.
- `include_qualified_name` (set to either "true" or "false", not required): include the qualified name of the event in the response body. Defaults to true.

**Example request**

```bash
curl http://localhost:80/events/someUniqueId123?sse=false&acquire_timeout=1&include_internal=false&include_qualified_name=true
```

**Successful response (`200 OK`)**

Single event payload:

```json
{
  "value": {"result": 12},
  "qualified_name": "__main__.MathEvent",
  "type": "__main__.MathEvent",
  "types": ["workflows.events.Event", "__main__.MathEvent"],
}
```

**Important considerations**

- Only one reader is allowed to stream the events per workflow run
- Once the events have been streamed, they cannot be recovered (unless you implemented some persistence logic on the client side)

> _We are working to improve both these aspects, so changes in the server behavior might be expected_

## Getting the result from a workflow execution (`/results/{handler_id}`)

> _This endpoint only works if you previously started a workflow asynchronously with `/run-nowait`_

To get the result of a previously started asynchronous workflow run, you can use the `/results/{handler_id}` endpoint passing the handler ID of the run.

**Example request**

```bash
curl http://localhost:80/results/someUniqueId123
```

**Successful response (`200 OK`)**

```json
{
  "handler_id": "someUniqueId123",
  "workflow_name": "math_workflow",
  "run_id": "uniqueRunId456",
  "error": null,
  "result": {
    "sum": 15,
    "subtraction": 9,
    "multiplication": 36,
    "division": 4,
  },
  "status": "completed",
  "started_at": "2024-10-21T14:32:15.123Z",
  "updated_at": "2024-10-21T14:45:30.456Z",
  "completed_at": "2024-10-21T14:45:30.456Z"
}
```

**Accepted response (`202 ACCEPTED`)**

Status code `202` is returned when the workflow is still running, and thus has not produce a result yet.

## Sending an event (`POST /events/{handler_id}`)

In cases where external input is needed for the workflow to run (human in the loop, e.g.), you can send a POST request to the `events/{handler_id}` endpoint with the event data to send (and, optionally, the step of the workflow to send them to) in order to provide said external input.

**Request body**

```json
{
  "event": {"__is_pydantic": true, "value": {"feedback": "This is great!", "approved": true}, "qualified_name": "__main__.HumanFeedbackEvent"},
  "step": "process_human_feedback"
}
```

- `event`: serialized representation of a workflow Event.
- `step` (optional): name of the step to send the event to.

**Successful response (`200 OK`)**

```json
{
  "status": "sent"
}
```

## Canceling a workflow run (`/handlers/{handler_id}/cancel`)

To stop a running workflow handler by cancelling its tasks, and optionally removing the associated handler from the persistence store, you can use `/handlers/{handler_id}/cancel`.

**Query parameters**

- `purge` (can be set to either "true" or "false", not required): whether or not to remove the handler associated with the workflow from the persistence store. Defaults to false.

**Example request**

```bash
curl -X POST http://localhost:80/handlers/someUniqueId123/cancel?purge=true
```

**Successful response (`200 OK`)**

```js
{
  "status": "deleted", // or canceled if purge is false
}
```

## Using `WorkflowClient` to Interact with Servers

In order to interact with a deployed `WorkflowServer` programmatically, beyond the raw API calls detailed above, we also provide a `WorkflowClient` class.

`WorkflowClient` provides methods for listing available workflows and workflow handlers, verifying the health of the server, running a workflow (both synchronously and asynchronously), streaming events and sending events.

Assuming you are running the server example from above, we can use `WorkflowClient` in the following way:

```python
from workflows.client import WorkflowClient

async def main()
  client = WorkflowClient(base_url="http://0.0.0.0:8080")
    workflows = await client.list_workflows()
    print("===== AVAILABLE WORKFLOWS ====")
    print(workflows)
    await client.is_healthy()  # will raise an exception if the server is not healthy
    handler = await client.run_workflow_nowait(
        "greet",
        start_event=StartEvent(name="John"),
        context=None,
    )
    handler_id = handler.handler_id
    print("==== STARTING THE WORKFLOW ===")
    print(f"Workflow running with handler ID: {handler_id}")
    print("=== STREAMING EVENTS ===")

    async for event in client.get_workflow_events(handler_id=handler_id):
        print("Received data:", event)
    result = await client.get_handler(handler_id)

    print(f"Final result: {result.result} (status: {result.status})")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

You can use the client also to interactively run human-in-the-loop workflows, like this one:

```python
from workflows import Workflow, step
from workflows.context import Context
from workflows.events import (
    StartEvent,
    StopEvent,
    InputRequiredEvent,
    HumanResponseEvent,
)
from workflows.server import WorkflowServer

class RequestEvent(InputRequiredEvent):
    prompt: str

class ResponseEvent(HumanResponseEvent):
    response: str

class OutEvent(StopEvent):
    output: str

class HumanInTheLoopWorkflow(Workflow):
    @step
    async def prompt_human(self, ev: StartEvent, ctx: Context) -> RequestEvent:
        return RequestEvent(prompt="What is your name?")

    @step
    async def greet_human(self, ev: ResponseEvent) -> OutEvent:
        return OutEvent(output=f"Hello, {ev.response}")

server = WorkflowServer()
server.add_workflow("human", HumanInTheLoopWorkflow(timeout=1000))
await server.serve("0.0.0.0", "8080")
```

You can now run the workflow and, when the human interaction is required, send the human response back:

```python
from workflow.client import WorkflowClient

client = WorkflowClient(base_url="http://0.0.0.0:8080")
handler = await client.run_workflow_nowait("human")
handler_id = handler.handler_id
print(handler_id)
async for event in client.get_workflow_events(handler_id=handler_id):
    if "RequestEvent" == event.type
        print(
            "Workflow is requiring human input:",
            event.value.get("prompt", ""),
        )
        name = input("Reply here: ")
        sent_event = await client.send_event(
            handler_id=handler_id,
            event=ResponseEvent(response=name.capitalize().strip()),
        )
        msg = "Event has been sent" if sent_event else "Event failed to send"
        print(msg)
result = await client.get_handler(handler_id)
print(f"Workflow complete with status: {result.status})")
res = OutEvent.model_validate(result.result)
print("Received final message:", res.output)
```
