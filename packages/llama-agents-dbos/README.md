# LlamaAgents DBOS Runtime

DBOS durable runtime plugin for LlamaIndex Workflows.

## Installation

```bash
pip install llama-agents-dbos
```

## Usage

```python
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

# Launch runtime and run workflow
runtime.launch()
result = await workflow.run()
```

## Features

- Durable workflow execution backed by DBOS
- Automatic step recording and replay
- Distributed workers and recovery support
