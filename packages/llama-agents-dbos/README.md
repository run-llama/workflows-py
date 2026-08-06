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

The runtime submits every workflow run through a stable DBOS queue named
`_llamaindex_workflow_queue:<workflow_name>`. The name is fixed when DBOS
registers the workflow. A later WorkflowServer route alias does not replace the
queue or strand existing work. The default remains unlimited:

```python
workflow = MyWorkflow(runtime=runtime, num_concurrent_runs=None)
```

Set `num_concurrent_runs` to a positive integer to limit active runs of that
workflow on each DBOS worker. Changing the value back to `None` removes the
limit without changing queues, so existing work remains available. Queue
admission normally takes about the configured
`DBOSRuntime(polling_interval_sec=...)`, but DBOS can back off longer while a
queue is idle.

Concurrency changes within the same DBOS application version are safe during a
rolling deployment. Workers can briefly use different limits while both
revisions are running, so total capacity is the sum of their per-worker limits.
The first deployment that moves existing direct-start runs onto queues can also
temporarily exceed the new limit while those earlier runs finish. If a
deployment changes the DBOS application version, keep workers for the old
version running until its queued and active workflows have drained.

Graceful cancellation remains event based. Cancelling an `ENQUEUED` run stores
the cancellation request, but the run must be admitted before it can publish a
`WorkflowCancelledEvent` and finish.

If the application calls `DBOS.listen_queues`, it must include every queue in
`runtime.workflow_queues`. The runtime raises an error before launch or late
workflow registration when a required queue is missing.

## Features

- Durable workflow execution backed by DBOS
- Automatic step recording and replay
- Distributed workers and recovery support
- Per-worker workflow concurrency with an unlimited default
