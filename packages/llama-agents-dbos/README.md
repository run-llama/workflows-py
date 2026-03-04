# LlamaAgents DBOS Runtime

DBOS durable runtime plugin for LlamaIndex Workflows.

## Installation

```bash
pip install llama-agents-dbos
```

## Usage

```python
import asyncio
from llama_agents.dbos import DBOSRuntime
from dbos import DBOS, DBOSConfig
from workflows import Workflow, step, StartEvent, StopEvent

# Configure DBOS
config: DBOSConfig = {
    "name": "my-app",
    "system_database_url": "postgresql://...",
}
DBOS(config=config)

# Create runtime and workflow
runtime = DBOSRuntime()

class MyWorkflow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="done")

workflow = MyWorkflow(runtime=runtime)

async def main():
    await runtime.launch()
    result = await workflow.run()

asyncio.run(main())
```

If you're not in an async context, use `runtime.launch_sync()` instead.

## Features

- Durable workflow execution backed by DBOS
- Automatic step recording and replay
- Distributed workers and recovery support
