# Server Examples

Expose workflows as HTTP APIs with `llama_agents.server.WorkflowServer`. Run standalone or mount inside an existing FastAPI / Starlette app.

## Examples

| File | What it shows |
| --- | --- |
| [`server_example.py`](server_example.py) | The minimal standalone `WorkflowServer`. **Start here.** |
| [`server_example.ipynb`](server_example.ipynb) | Same example as a walkthrough notebook, with explanations and a client call. |
| [`server.py`](server.py) | Mount `WorkflowServer` inside an existing FastAPI app alongside your own routes. |
| [`fastapi_server_example.ipynb`](fastapi_server_example.ipynb) | Notebook version of the FastAPI integration pattern. |

## Running

```bash
uv run examples/server/server_example.py
# then in another terminal
curl -X POST http://localhost:8000/workflows/echo/run -d '{"start_event": {"message": "hi"}}'
```

See also the [`client/`](../client/) examples for calling a running server from Python.
