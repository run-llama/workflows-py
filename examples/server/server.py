import uvicorn
from fastapi import FastAPI
from llama_agents.server import WorkflowServer
from pydantic import BaseModel
from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent


class UserModel(BaseModel):
    name: str
    email: str


# Existing FastAPI application with some routes
app = FastAPI(title="My API with Workflows", version="1.0.0")


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


def main() -> None:
    # Create workflow server
    workflow_server = WorkflowServer()

    # Register workflows
    workflow_server.add_workflow("user_processing", UserProcessingWorkflow())
    workflow_server.add_workflow("notification", NotificationWorkflow())

    # Mount workflow server as sub-application
    app.mount("/wf-server", workflow_server.app)

    # run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
