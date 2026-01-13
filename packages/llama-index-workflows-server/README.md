# LlamaIndex Workflows Server

HTTP server for serving [LlamaIndex Workflows](https://pypi.org/project/llama-index-workflows/) as web services.

This package provides:

- **WorkflowServer**: HTTP server for serving workflows as web services
- **AbstractWorkflowStore**: Interface for workflow handler persistence
- **SqliteWorkflowStore**: SQLite-based persistence implementation

## Installation

```bash
pip install llama-index-workflows-server
```

Or via the main workflows package:

```bash
pip install llama-index-workflows[server]
```

## Usage

```python
from workflows import Workflow, Context, step
from workflows.events import StartEvent, StopEvent
from workflows.server import WorkflowServer

class MyWorkflow(Workflow):
    @step
    async def process(self, ctx: Context, ev: StartEvent) -> StopEvent:
        return StopEvent(result="done")

server = WorkflowServer()
server.add(MyWorkflow, name="my-workflow")
server.run()
```

## Client

The workflow client is included in the main `llama-index-workflows` package:

```bash
pip install llama-index-workflows[client]
```

```python
from workflows.client import WorkflowClient

client = WorkflowClient(base_url="http://localhost:8000")
result = await client.run_workflow("my-workflow")
```

## License

MIT
