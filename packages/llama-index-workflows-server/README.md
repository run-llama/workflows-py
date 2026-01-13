# LlamaIndex Workflows Server

Server, client and protocol components for [LlamaIndex Workflows](https://pypi.org/project/llama-index-workflows/).

This package provides:

- **WorkflowServer**: HTTP server for serving workflows as web services
- **WorkflowClient**: Async HTTP client for interacting with workflow servers
- **Protocol**: Shared data models for client-server communication

## Installation

```bash
# Install with server dependencies
pip install llama-index-workflows-server[server]

# Install with client dependencies
pip install llama-index-workflows-server[client]

# Install both
pip install llama-index-workflows-server[server,client]
```

Or via the main workflows package:

```bash
pip install llama-index-workflows[server]
pip install llama-index-workflows[client]
```

## Usage

### Server

```python
from workflows import Workflow, Context, step
from workflows.server import WorkflowServer

class MyWorkflow(Workflow):
    @step
    async def process(self, ctx: Context, ev: StartEvent) -> StopEvent:
        return StopEvent(result="done")

server = WorkflowServer()
server.add(MyWorkflow, name="my-workflow")
server.run()
```

### Client

```python
from workflows.client import WorkflowClient

client = WorkflowClient(base_url="http://localhost:8000")
result = await client.run_workflow("my-workflow")
```

## License

MIT
