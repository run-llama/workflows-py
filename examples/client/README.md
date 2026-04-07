# Client Examples

Call a running `WorkflowServer` from Python with `llama_agents.client.WorkflowClient`. Each subfolder is a self-contained pair: a server script you run first, and a client script you run against it.

## Examples

| Folder | What it shows |
| --- | --- |
| [`base/`](base/) | The minimal client/server pair — submit a run, stream events, get the result. **Start here.** |
| [`human_in_the_loop/`](human_in_the_loop/) | A workflow that pauses to wait for human input via `InputRequiredEvent` / `HumanResponseEvent`, and a client that responds to it. |

## Running

In one terminal, start the server:

```bash
uv run examples/client/base/workflow_server.py
```

In another, run the client:

```bash
uv run examples/client/base/workflow_client.py
```
