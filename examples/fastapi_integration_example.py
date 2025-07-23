#!/usr/bin/env python3
"""
Example demonstrating how to integrate WorkflowServer into an existing FastAPI application.

This example shows how to:
1. Create a FastAPI application with existing routes
2. Set up WorkflowServer with workflows
3. Mount the workflow server as a sub-application
4. Run the combined application
"""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent
from workflows.server import WorkflowServer

# Existing FastAPI application with some routes
app = FastAPI(title="My API with Workflows", version="1.0.0")


class UserModel(BaseModel):
    name: str
    email: str


# Existing API routes
@app.get("/")
async def root() -> dict:
    return {"message": "Welcome to My API"}


@app.get("/users/{user_id}")
async def get_user(user_id: int) -> dict:
    return {
        "user_id": user_id,
        "name": f"User {user_id}",
        "email": f"user{user_id}@example.com",
    }


@app.post("/users")
async def create_user(user: UserModel) -> dict:
    return {"message": f"Created user {user.name}", "user": user}


# Define workflows
class UserProcessingWorkflow(Workflow):
    @step
    async def process_user(self, ev: StartEvent) -> StopEvent:
        user_data = getattr(ev, "user_data", {})
        name = user_data.get("name", "Unknown")
        email = user_data.get("email", "unknown@example.com")

        # Simulate some processing
        processed_data = {
            "processed_name": name.upper(),
            "domain": email.split("@")[1] if "@" in email else "unknown",
            "status": "processed",
        }

        return StopEvent(result=processed_data)


class NotificationWorkflow(Workflow):
    @step
    async def send_notification(self, ev: StartEvent) -> StopEvent:
        message = getattr(ev, "message", "Default notification")
        recipient = getattr(ev, "recipient", "admin@example.com")

        # Simulate sending notification
        result = {
            "notification_id": "notif_123",
            "message": message,
            "recipient": recipient,
            "sent_at": "2024-01-01T12:00:00Z",
            "status": "sent",
        }

        return StopEvent(result=result)


def setup_app() -> FastAPI:
    """Set up the complete application with workflows."""
    # Create workflow server
    workflow_server = WorkflowServer()

    # Register workflows
    workflow_server.add_workflow("user_processing", UserProcessingWorkflow())
    workflow_server.add_workflow("notification", NotificationWorkflow())

    # Mount workflow server as sub-application
    app.mount("/workflows", workflow_server.app)

    return app


def main() -> None:
    """Run the server."""
    complete_app = setup_app()

    print("Starting FastAPI application with workflows on http://localhost:8000")
    print("\nExisting API endpoints:")
    print("  GET  / - Root endpoint")
    print("  GET  /users/{user_id} - Get user by ID")
    print("  POST /users - Create user")
    print("\nWorkflow endpoints (mounted at /workflows):")
    print("  GET  /workflows/health - Health check")
    print("  GET  /workflows/workflows - List all workflows")
    print("  GET  /workflows/workflows/{name} - Get workflow info")
    print("  POST /workflows/workflows/{name}/run - Run workflow synchronously")
    print("  POST /workflows/workflows/{name}/run-async - Run workflow asynchronously")

    uvicorn.run(complete_app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()


# Example HTTP requests you can make:
"""
# Existing API routes
curl http://localhost:8000/
curl http://localhost:8000/users/123
curl -X POST http://localhost:8000/users \
  -H "Content-Type: application/json" \
  -d '{"name": "John Doe", "email": "john@example.com"}'

# Workflow routes (note the /workflows prefix)
curl http://localhost:8000/workflows/health
curl http://localhost:8000/workflows/workflows

# Run user processing workflow
curl -X POST http://localhost:8000/workflows/workflows/user_processing/run \
  -H "Content-Type: application/json" \
  -d '{"kwargs": {"user_data": {"name": "alice", "email": "alice@company.com"}}}'

# Run notification workflow
curl -X POST http://localhost:8000/workflows/workflows/notification/run \
  -H "Content-Type: application/json" \
  -d '{"kwargs": {"message": "Welcome to our platform!", "recipient": "user@example.com"}}'

# Run workflow asynchronously
curl -X POST http://localhost:8000/workflows/workflows/user_processing/run-async \
  -H "Content-Type: application/json" \
  -d '{"kwargs": {"user_data": {"name": "bob", "email": "bob@startup.io"}}}'
"""
