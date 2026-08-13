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

# launch_sync() works outside async contexts; use await runtime.launch() inside one
runtime.launch_sync()

async def main():
    result = await workflow.run()

asyncio.run(main())
```

## Workflow concurrency

Set `num_concurrent_runs` to limit how many runs of a workflow may be active
at once on each DBOS worker:

```python
workflow = MyWorkflow(runtime=runtime, num_concurrent_runs=8)
```

The default is `None`, which is unlimited. Unlimited workflows start directly,
with no queue in the path. Limited workflows submit through a DBOS queue named
`_llamaindex_workflow_queue:<workflow_name>`, and runs beyond the limit wait as
`ENQUEUED`. Admission takes about the configured
`DBOSRuntime(polling_interval_sec=...)`, one second by default. Capacity across
a deployment is the limit times the number of workers. The queue is shared, so
an enqueued run has no affinity to the replica that submitted it. Any worker
with a free slot can pick it up.

The runtime declares the queue for every workflow, limited or not, so turning a
limit on or off never strands queued work. A new limit does not count runs that
started before it, so a worker can briefly exceed the limit while those finish.

Waiting runs are rows in the database, filed under the workflow's name
(`workflow_name`, defaulting to the Python module and class name). A worker
only looks for waiting work under the names it knows, so if you rename a
workflow, rows filed under the old name are invisible to the new deployment.
Keep old workers running until they finish that work.

A run that is still waiting in the queue cannot be cancelled yet, because
cancellation is a message delivered to the running workflow. The request is
saved, and the run stops itself as soon as it starts.

DBOS normally watches every queue automatically. An application that instead
passes an explicit list to `DBOS.listen_queues` must add this runtime's
queues to it (`runtime.workflow_queues`), or waiting runs are never picked
up. Build the list after registering workflows and before launch.

## Features

- Durable workflow execution backed by DBOS
- Automatic step recording and replay
- Distributed workers and recovery support
- Per-worker workflow concurrency with an unlimited default
