#!/usr/bin/env python3
"""
Example demonstrating how to use the WorkflowServer.

This example shows how to:
1. Create a simple workflow
2. Set up the server
3. Register workflows
4. Run the server
5. Make HTTP requests to execute workflows
"""

import asyncio

from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent
from workflows.server import WorkflowServer


# Define a simple workflow
class GreetingWorkflow(Workflow):
    @step
    async def greet(self, ev: StartEvent) -> StopEvent:
        name = getattr(ev, "name", "World")
        return StopEvent(result=f"Hello, {name}!")


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


async def main() -> None:
    """Run the server."""
    server = WorkflowServer()

    # Register workflows
    server.add_workflow("greeting", GreetingWorkflow())
    server.add_workflow("math", MathWorkflow())

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
# 1. Start long-running workflow
curl -X POST http://localhost:8000/workflows/greeting/run-nowait \
  -H "Content-Type: application/json" \
  -d '{"kwargs": {"name": "Async User"}}'

# 2. Get the handler_id from response and check result
# curl http://localhost:8000/results/{handler_id}
"""
