from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent


class FileEvent(StartEvent):
    file_id: str


class Metadata(BaseModel):
    name: str


class DummyWorkflow(Workflow):
    @step
    async def take_step(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="hello")


class DummyFileWorkflow(Workflow):
    @step
    async def take_step(self, ev: FileEvent) -> StopEvent:
        return StopEvent(result=ev.file_id)


class DummyMetadataWorkflow(Workflow):
    @step
    async def take_step(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result=Metadata(name="nemo"))


class DummyWorkflowWithError(Workflow):
    @step
    async def take_step(self, ev: StartEvent) -> StopEvent:
        raise ValueError("You shall not pass!")


@dataclass
class MockContext:
    session_id: str = "session-1"


@dataclass
class MockHandlerData:
    handler_id: str = "handler-1"
    workflow_name: str = "default"
    run_id: str = "run-1"
    status: str = "completed"
    result: dict | None = None
    error: str | None = None
    started_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None


class MockAgentCoreService:
    def __init__(self, with_error: bool = False) -> None:
        self.return_handler_data = MockHandlerData(result={"hello": "world"})
        self.call_args: list[dict[str, Any]] = []
        self.with_error = with_error

    async def run_workflow(
        self,
        workflow_name: str,
        *,
        start_event: StartEvent,
    ) -> MockHandlerData:
        self.call_args.append(
            {"workflow_name": workflow_name, "start_event": start_event}
        )
        if workflow_name == "with_error":
            raise ValueError("You shall not pass!")
        if not self.with_error:
            return self.return_handler_data
        return MockHandlerData(result={}, error="Some fancy error")

    async def run_workflow_with_session(
        self,
        workflow_name: str,
        start_event: StartEvent,
        handler_id: str,
        *,
        nowait: bool = False,
    ) -> MockHandlerData:
        self.call_args.append(
            {
                "workflow_name": workflow_name,
                "start_event": start_event,
                "handler_id": handler_id,
                "nowait": nowait,
            }
        )
        if workflow_name == "with_error":
            raise ValueError("You shall not pass!")
        if not self.with_error:
            data = MockHandlerData(
                handler_id=handler_id,
                workflow_name=workflow_name,
                result={"hello": "world"},
            )
            return data
        return MockHandlerData(
            handler_id=handler_id,
            workflow_name=workflow_name,
            result={},
            error="Some fancy error",
        )

    def get_workflow_names(self) -> list[str]:
        return ["default", "metadata", "process-file"]

    async def get_handler(self, handler_id: str) -> MockHandlerData | None:
        return None

    async def query_handlers(self, query: Any) -> list:
        return []

    async def get_events(
        self,
        handler_id: str,
        after_sequence: int | None = None,
        limit: int | None = None,
    ) -> list:
        return []

    async def send_event(
        self, handler_id: str, event: Any, step: str | None = None
    ) -> None:
        pass

    async def cancel_handler(self, handler_id: str, purge: bool = False) -> str | None:
        return "cancelled"


class MockBedrockApp:
    def __init__(self) -> None:
        self.tasks: dict[int, str] = {}
        self.added = 0
        self.completed = 0

    def add_async_task(self, name: str, metadata: dict[str, Any] | None = None) -> int:
        self.added += 1
        self.tasks[id(name)] = name
        return id(name)

    def complete_async_task(self, task_id: int) -> bool:
        self.completed += 1
        if task_id in self.tasks:
            self.tasks.pop(task_id)
            return True
        return False
