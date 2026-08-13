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

The queue name comes from `workflow_name`, which defaults to the Python module
and class name. Treat it as a durable identifier. Renaming it also renames the
DBOS registrations, so workers using the old name must drain their work before
they are removed.

Cancelling an `ENQUEUED` run takes effect after admission, because cancellation
is delivered as an event to the running control loop.

Applications that restrict `DBOS.listen_queues` must include every queue in
`runtime.workflow_queues`, collected after registering workflows and before
launch.

## Features

- Durable workflow execution backed by DBOS
- Automatic step recording and replay
- Distributed workers and recovery support
- Per-worker workflow concurrency with an unlimited default
