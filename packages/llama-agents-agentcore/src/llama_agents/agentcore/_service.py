from bedrock_agentcore.runtime import BedrockAgentCoreApp
from llama_agents.client import HandlerData
from llama_agents.server._runtime.idle_release_runtime import IdleReleaseDecorator
from llama_agents.server._runtime.persistence_runtime import PersistenceDecorator
from llama_agents.server._runtime.server_runtime import ServerRuntimeDecorator
from llama_agents.server._service import _WorkflowService as WorkflowService
from llama_agents.server._store.abstract_workflow_store import (
    AbstractWorkflowStore,
    HandlerQuery,
    StoredEvent,
)
from workflows import Workflow
from workflows.events import Event, StartEvent
from workflows.plugins.basic import basic_runtime
from workflows.runtime.types.plugin import Runtime
from workflows.utils import _nanoid

from ._runtime_decorator import AgentCoreRuntimeDecorator

IDLE_TIMEOUT = 60.0
PERSISTENCE_BACKOFF = [0.5, 3]
POLLING_TIMEOUT = 600


class WorkflowNotFoundError(Exception):
    """
    Raise when a workflow with a given name cannot be found within the registered workflows.
    """

    def __init__(self, workflow_name: str) -> None:
        self.workflow_name = workflow_name

    def __repr__(self) -> str:
        return f"Could not find {self.workflow_name} among the registered workflows"

    def __str__(self) -> str:
        return f"Could not find {self.workflow_name} among the registered workflows"


class AgentCoreService:
    def __init__(
        self,
        app: BedrockAgentCoreApp,
        store: AbstractWorkflowStore,
        idle_timeout: float = IDLE_TIMEOUT,
        persistence_backoff: list[float] = PERSISTENCE_BACKOFF,
    ) -> None:
        self._workflow_store = store
        inner: Runtime = IdleReleaseDecorator(
            PersistenceDecorator(basic_runtime, store=self._workflow_store),
            store=self._workflow_store,
            idle_timeout=idle_timeout,
        )
        server: ServerRuntimeDecorator = ServerRuntimeDecorator(
            inner,
            store=self._workflow_store,
            persistence_backoff=persistence_backoff,
        )
        self._runtime: AgentCoreRuntimeDecorator = AgentCoreRuntimeDecorator(
            decorated=server,
            store=self._workflow_store,
            app=app,
            persistence_backoff=persistence_backoff,
        )
        self._service = WorkflowService(
            runtime=self._runtime, store=self._workflow_store
        )

    def add_workflow(self, workflow_name: str, workflow: Workflow) -> None:
        workflow._switch_workflow_name(workflow_name)
        workflow._switch_runtime(self._runtime)

    def get_workflow(self, workflow_name: str) -> Workflow | None:
        """Get a registered workflow by name."""
        return self._runtime.get_workflow(workflow_name)

    def get_workflow_names(self) -> list[str]:
        """Get all registered workflow names."""
        return self._runtime.get_workflow_names()

    async def run_workflow(
        self, workflow_name: str, start_event: StartEvent
    ) -> HandlerData:
        wf = self._service.get_workflow(workflow_name)
        if wf is not None:
            handler_data = await self._service.start_workflow(
                wf, start_event=start_event, handler_id=_nanoid()
            )
            return await self._service.await_workflow(handler_data)

        raise WorkflowNotFoundError(workflow_name)

    async def run_workflow_with_session(
        self,
        workflow_name: str,
        start_event: StartEvent,
        handler_id: str,
        *,
        nowait: bool = False,
    ) -> HandlerData:
        """Run a workflow using a specific handler_id (typically the session ID).

        If a handler already exists for this ID:
        - completed → return cached result
        - running   → await it (sync) or return current status (nowait)
        - failed/cancelled → start fresh

        This makes invocations idempotent within a session.
        """
        # Check for existing handler
        existing = await self._service.load_handler(handler_id)
        if existing is not None:
            status = existing.status
            if status == "completed":
                return existing
            if status == "running":
                if nowait:
                    return existing
                return await self._service.await_workflow(existing)
            # failed/cancelled → fall through to start fresh

        wf = self._service.get_workflow(workflow_name)
        if wf is None:
            raise WorkflowNotFoundError(workflow_name)

        handler_data = await self._service.start_workflow(
            wf, start_event=start_event, handler_id=handler_id
        )
        if nowait:
            return handler_data
        return await self._service.await_workflow(handler_data)

    async def get_handler(self, handler_id: str) -> HandlerData | None:
        """Get handler status and result by ID."""
        return await self._service.load_handler(handler_id)

    async def query_handlers(self, query: HandlerQuery) -> list:
        """Query handlers with optional filters."""
        return await self._service.query_handlers(query)

    async def get_events(
        self,
        handler_id: str,
        after_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[StoredEvent]:
        """Get recorded events for a handler."""
        handler = await self._service.load_handler(handler_id)
        if handler is None:
            return []
        run_id = getattr(handler, "run_id", None)
        if not run_id:
            return []
        return await self._workflow_store.query_events(
            run_id, after_sequence=after_sequence, limit=limit
        )

    async def send_event(
        self, handler_id: str, event: Event, step: str | None = None
    ) -> None:
        """Send an event into a running workflow (human-in-the-loop)."""
        await self._service.send_event(handler_id, event, step=step)

    async def cancel_handler(self, handler_id: str, purge: bool = False) -> str | None:
        """Cancel a running workflow handler."""
        return await self._service.cancel_handler(handler_id, purge=purge)
