from bedrock_agentcore.runtime import BedrockAgentCoreApp
from llama_agents.server._runtime.server_runtime import ServerRuntimeDecorator
from llama_agents.server._store.abstract_workflow_store import AbstractWorkflowStore
from workflows import Workflow
from workflows.events import Event
from workflows.runtime.types.plugin import (
    RegisteredWorkflow,
    Runtime,
)
from workflows.runtime.types.results import StepFunctionResult, StepWorkerState
from workflows.runtime.types.step_function import (
    StepWorkerFunction,
    as_step_worker_functions,
    create_workflow_run_function,
)


def as_agentcore_async_task(
    app: BedrockAgentCoreApp, name: str, fn: StepWorkerFunction
) -> StepWorkerFunction:
    async def step_worker(
        state: StepWorkerState,
        step_name: str,
        event: Event,
        workflow: Workflow,
    ) -> list[StepFunctionResult]:
        task_id = app.add_async_task(name)
        try:
            results = await fn(state, step_name, event, workflow)
        finally:
            app.complete_async_task(task_id)
        return results

    return step_worker


class AgentCoreRuntimeDecorator(ServerRuntimeDecorator):
    def __init__(
        self,
        decorated: Runtime,
        store: AbstractWorkflowStore,
        app: BedrockAgentCoreApp,
        *,
        persistence_backoff: list[float] | None = None,
    ) -> None:
        super().__init__(decorated, store, persistence_backoff=persistence_backoff)
        self.app = app
        self._tracked: dict[str, Workflow] = {}
        self._registered: dict[int, RegisteredWorkflow] = {}

    def register(self, workflow: Workflow) -> RegisteredWorkflow:
        name = workflow.workflow_name
        if name not in self._tracked:
            self._tracked[name] = workflow
        wrapped_steps: dict[str, StepWorkerFunction] = {
            step_name: as_agentcore_async_task(self.app, f"{name}.{step_name}", step)
            for step_name, step in as_step_worker_functions(workflow).items()
        }
        registered = RegisteredWorkflow(
            steps=wrapped_steps,
            workflow_run_fn=create_workflow_run_function(workflow),
            workflow=workflow,
        )
        id_ = id(workflow)
        self._registered[id_] = registered
        return registered

    def track_workflow(self, workflow: Workflow) -> None:
        self._tracked[workflow.workflow_name] = workflow
        super().track_workflow(workflow)

    def untrack_workflow(self, workflow: Workflow) -> None:
        self._tracked.pop(workflow.workflow_name, None)
        super().untrack_workflow(workflow)

    def get_registered(self, workflow: Workflow) -> RegisteredWorkflow | None:
        return self._registered.get(id(workflow))

    def get_workflow(self, name: str) -> Workflow | None:
        return self._tracked.get(name)

    def get_workflow_names(self) -> list[str]:
        return list(self._tracked.keys())
