# Deploying a Workflow

## Deploying with `WorkflowServer`

The `workflows` library includes a `WorkflowServer` class that allows you to easily expose your workflows over an HTTP
API. This provides a flexible way to run and manage workflows from any HTTP-capable client.

### Programmatic Usage

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

### Command-Line Interface (CLI)

The library also provides a convenient CLI to run a server from a file containing a `WorkflowServer` instance.

Given the `my_server.py` file from the example above, you can start the server with the following command:

```bash
python -m workflows.server my_server.py
```

The server will start on `0.0.0.0:8080` by default. You can configure the host and port using the
`WORKFLOWS_PY_SERVER_HOST` and `WORKFLOWS_PY_SERVER_PORT` environment variables.

### API Endpoints

The `WorkflowServer` exposes the following RESTful endpoints:

| Method | Path                           | Description                                                                                             |
|--------|--------------------------------|---------------------------------------------------------------------------------------------------------|
| `GET`  | `/health`                      | Returns a health check response (`{"status": "healthy"}`).                                               |
| `GET`  | `/workflows`                   | Lists the names of all registered workflows.                                                            |
| `POST` | `/workflows/{name}/run`        | Runs the specified workflow synchronously and returns the final result.                                 |
| `POST` | `/workflows/{name}/run-nowait` | Starts the specified workflow asynchronously and returns a `handler_id`.                                |
| `GET`  | `/results/{handler_id}`        | Retrieves the result of an asynchronously run workflow. Returns `202 Accepted` if still running.       |
| `GET`  | `/events/{handler_id}`         | Streams all events from a running workflow as newline-delimited JSON (`application/x-ndjson`).          |

#### Running a Workflow (`/run`)

To run a workflow and wait for its completion, send a `POST` request to `/workflows/{name}/run`.

**Request Body:**

```json
{
  "kwargs": {
    "a": 5,
    "b": 10
  }
}
```

**Successful Response (`200 OK`):**

```json
{
  "result": 15
}
```

#### Running a Workflow Asynchronously (`/run-nowait`)

To start a workflow without waiting for it to finish, use the `/run-nowait` endpoint.

**Request Body:**

```json
{
  "kwargs": {
    "a": 5,
    "b": 10
  }
}
```

**Successful Response (`200 OK`):**

```json
{
  "handler_id": "someUniqueId123",
  "status": "started"
}
```

You can then use the `handler_id` to check for the result or stream events.
