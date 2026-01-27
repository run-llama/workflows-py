---
sidebar:
  order: 15
title: Working with blocking code
---

Workflows are built on top of Python's `asyncio` event loop. Every workflow step is an `async` function, and the runtime relies on cooperative multitasking: while one step is `await`-ing (for example, waiting for an LLM response or a network call), other steps and workflows are free to make progress.

**Blocking code breaks this model.** If a step calls a synchronous function that takes a long time to return — whether it's doing I/O or heavy computation — the entire event loop stalls. No other steps, workflows, or even unrelated async tasks can run until that blocking call finishes.

This page covers how to safely integrate blocking code into your workflows.

## Blocking I/O libraries

Many Python libraries only offer synchronous APIs: database drivers, HTTP clients, file system operations, SDK calls, and more. When you need to use one of these inside a workflow step, offload the call to a thread pool using `asyncio.to_thread`:

```python
import asyncio
import requests
from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent


class BlockingIOWorkflow(Workflow):
    @step
    async def fetch_data(self, ev: StartEvent) -> StopEvent:
        # Bad: this blocks the event loop until the request completes
        # response = requests.get("https://api.example.com/data")

        # Good: run the blocking call in a thread so the event loop stays free
        response = await asyncio.to_thread(
            requests.get, "https://api.example.com/data"
        )
        return StopEvent(result=response.json())
```

`asyncio.to_thread` schedules the function on the default `ThreadPoolExecutor` and returns an awaitable. While the blocking call runs in a separate thread, the event loop continues processing other steps and workflows.

This applies to any synchronous library call that performs I/O — reading files, querying databases, calling external APIs, and so on:

```python
import asyncio
import json
from pathlib import Path
from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent


def read_large_file(path: str) -> dict:
    """A synchronous function that reads and parses a large JSON file."""
    return json.loads(Path(path).read_text())


class FileReaderWorkflow(Workflow):
    @step
    async def process_file(self, ev: StartEvent) -> StopEvent:
        data = await asyncio.to_thread(read_large_file, ev.file_path)
        return StopEvent(result=data)
```

## CPU-intensive operations

CPU-bound work — data transformation, image processing, numerical computation — presents a different challenge. Even when run on a thread, CPU-intensive Python code can contend with the event loop due to the GIL (Global Interpreter Lock).

For CPU-heavy work, use a dedicated, smaller thread pool (or process pool) so that these tasks are queued and do not saturate the default executor:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent


# A small, dedicated pool for CPU-bound work.
# Keeping this small ensures CPU tasks are queued rather than
# overwhelming the system with parallel CPU-bound threads.
cpu_pool = ThreadPoolExecutor(max_workers=2)


def expensive_computation(data: str) -> str:
    """A CPU-intensive operation, e.g. data parsing or transformation."""
    # Simulate heavy work
    result = data
    for _ in range(1_000_000):
        result = result.strip()
    return result


class ComputeEvent(Event):
    data: str


class CPUWorkflow(Workflow):
    @step
    async def start(self, ev: StartEvent) -> ComputeEvent:
        return ComputeEvent(data=ev.input_data)

    @step
    async def compute(self, ev: ComputeEvent) -> StopEvent:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(cpu_pool, expensive_computation, ev.data)
        return StopEvent(result=result)
```

Key differences from the I/O case:

- **Use `loop.run_in_executor`** with an explicit executor instead of `asyncio.to_thread` so you can control the pool size and type.
- **Keep the pool small.** A pool of 1–2 workers means CPU tasks queue up rather than competing for CPU time. Adjust based on your workload and available cores.
- **Consider a `ProcessPoolExecutor`** for truly CPU-bound work that you want to run outside the GIL. The API is the same — just swap the executor type:

```python
from concurrent.futures import ProcessPoolExecutor

cpu_pool = ProcessPoolExecutor(max_workers=2)
```

Note that functions submitted to a `ProcessPoolExecutor` must be picklable (top-level functions, not lambdas or closures).

## Summary

| Scenario | Solution | Why |
|---|---|---|
| Synchronous I/O (HTTP, DB, file) | `await asyncio.to_thread(fn, ...)` | Frees the event loop while I/O completes in a thread |
| CPU-intensive work | `await loop.run_in_executor(pool, fn, ...)` with a small dedicated pool | Queues heavy computation so it doesn't starve the event loop or other tasks |

The core principle is straightforward: **never let a workflow step block the asyncio event loop.** Any work that isn't natively async should be offloaded to a thread or process and `await`-ed, so the runtime can continue processing other steps and workflows concurrently.
