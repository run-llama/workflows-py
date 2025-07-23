#!/usr/bin/env python3
"""
Example demonstrating how to use the WorkflowServer with event streaming.

This example shows how to:
1. Create workflows that emit streaming events
2. Set up the server with event streaming support
3. Register workflows
4. Run the server
5. Make HTTP requests to execute workflows
6. Stream real-time events from running workflows using the /events endpoint
"""

import asyncio

from workflows import Workflow, step
from workflows.context import Context
from workflows.events import Event, StartEvent, StopEvent
from workflows.server import WorkflowServer


class StreamEvent(Event):
    sequence: int


# Define a simple workflow
class GreetingWorkflow(Workflow):
    @step
    async def greet(self, ctx: Context, ev: StartEvent) -> StopEvent:
        for i in range(3):
            ctx.write_event_to_stream(StreamEvent(sequence=i))
            await asyncio.sleep(0.3)

        name = getattr(ev, "name", "World")
        return StopEvent(result=f"Hello, {name}!")


class ProgressEvent(Event):
    step: str
    progress: int
    message: str


class MathWorkflow(Workflow):
    @step
    async def calculate(self, ev: StartEvent) -> StopEvent:
        a = getattr(ev, "a", 0)
        b = getattr(ev, "b", 0)
        operation = getattr(ev, "operation", "add")

        if operation == "add":
            result = a + b
        elif operation == "multiply":
            result = a * b
        elif operation == "subtract":
            result = a - b
        elif operation == "divide":
            result = a / b if b != 0 else None
        else:
            result = None

        return StopEvent(
            result={"a": a, "b": b, "operation": operation, "result": result}
        )


class ProcessingWorkflow(Workflow):
    """Example workflow that demonstrates event streaming with progress updates."""

    @step
    async def process(self, ctx: Context, ev: StartEvent) -> StopEvent:
        items = getattr(ev, "items", ["item1", "item2", "item3", "item4", "item5"])

        ctx.write_event_to_stream(
            ProgressEvent(
                step="start",
                progress=0,
                message=f"Starting processing of {len(items)} items",
            )
        )

        results = []
        for i, item in enumerate(items):
            # Simulate processing time
            await asyncio.sleep(0.5)

            # Emit progress event
            progress = int((i + 1) / len(items) * 100)
            ctx.write_event_to_stream(
                ProgressEvent(
                    step="processing",
                    progress=progress,
                    message=f"Processed {item} ({i + 1}/{len(items)})",
                )
            )

            results.append(f"processed_{item}")

        ctx.write_event_to_stream(
            ProgressEvent(
                step="complete",
                progress=100,
                message="Processing completed successfully",
            )
        )

        return StopEvent(result={"processed_items": results, "total": len(results)})


async def main() -> None:
    """Run the server."""
    server = WorkflowServer()

    # Register workflows
    server.add_workflow("greeting", GreetingWorkflow())
    server.add_workflow("math", MathWorkflow())
    server.add_workflow("processing", ProcessingWorkflow())

    await server.serve(host="0.0.0.0", port=8000)


if __name__ == "__main__":
    asyncio.run(main())


# Example HTTP requests you can make:
"""
# Health check
curl http://localhost:8000/health

# List workflows
curl http://localhost:8000/workflows

# Get workflow info
curl http://localhost:8000/workflows/greeting

# Run greeting workflow
curl -X POST http://localhost:8000/workflows/greeting/run \
  -H "Content-Type: application/json" \
  -d '{"kwargs": {"name": "Alice"}}'

# Run math workflow
curl -X POST http://localhost:8000/workflows/math/run \
  -H "Content-Type: application/json" \
  -d '{"kwargs": {"a": 10, "b": 5, "operation": "multiply"}}'

# Run workflow with context
curl -X POST http://localhost:8000/workflows/greeting/run \
  -H "Content-Type: application/json" \
  -d '{"context": {"user_id": "123"}, "kwargs": {"name": "Bob"}}'

# Run workflow asynchronously (nowait)
curl -X POST http://localhost:8000/workflows/math/run-nowait \
  -H "Content-Type: application/json" \
  -d '{"kwargs": {"a": 100, "b": 25, "operation": "divide"}}'

# Example response: {"handler_id": "abc123", "status": "started"}

# Get result of async workflow (replace {handler_id} with actual handler_id from above)
curl http://localhost:8000/results/abc123

# Example async workflow sequence:
# 1. Start long-running workflow with streaming events
curl -X POST http://localhost:8000/workflows/greeting/run-nowait \
  -H "Content-Type: application/json" \
  -d '{"kwargs": {"name": "Async User"}}'

# 2. Stream events from the workflow (replace {handler_id} with actual handler_id from above)
curl http://localhost:8000/events/{handler_id}

# 3. Stream raw events (includes event metadata)
curl http://localhost:8000/events/{handler_id}?raw_event=true

# 4. Get the final result after events complete
curl http://localhost:8000/results/{handler_id}

# Example event streaming sequence:
# 1. Start the greeting workflow asynchronously:
curl -X POST http://localhost:8000/workflows/greeting/run-nowait \
  -H "Content-Type: application/json" \
  -d '{"kwargs": {"name": "StreamDemo"}}'

# Response: {"handler_id": "xyz789", "status": "started"}

# 2. In another terminal, immediately start streaming events:
curl http://localhost:8000/events/xyz789

# You should see streaming output like:
# {"sequence": 0}
# {"sequence": 1}
# {"sequence": 2}

# 3. After events complete, get the final result:
curl http://localhost:8000/results/xyz789

# Response: {"result": "Hello, StreamDemo!"}

# Advanced event streaming example with progress tracking:
# 1. Start processing workflow:
curl -X POST http://localhost:8000/workflows/processing/run-nowait \
  -H "Content-Type: application/json" \
  -d '{"kwargs": {"items": ["task1", "task2", "task3"]}}'

# Response: {"handler_id": "proc123", "status": "started"}

# 2. Stream progress events in real-time:
curl http://localhost:8000/events/proc123

# You should see streaming output like:
# {"step": "start", "progress": 0, "message": "Starting processing of 3 items"}
# {"step": "processing", "progress": 33, "message": "Processed task1 (1/3)"}
# {"step": "processing", "progress": 66, "message": "Processed task2 (2/3)"}
# {"step": "processing", "progress": 100, "message": "Processed task3 (3/3)"}
# {"step": "complete", "progress": 100, "message": "Processing completed successfully"}

# 3. Get final result:
curl http://localhost:8000/results/proc123

# Response: {"result": {"processed_items": ["processed_task1", "processed_task2", "processed_task3"], "total": 3}}

# Event streaming with raw event data (includes event metadata):
curl http://localhost:8000/events/{handler_id}?raw_event=true

# This returns full event objects with metadata like event type, timestamp, etc.
"""
