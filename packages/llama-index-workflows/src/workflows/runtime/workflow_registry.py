from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Optional

from workflows.runtime.types._identity_weak_ref import IdentityWeakKeyDict
from workflows.runtime.types.plugin import (
    ControlLoopFunction,
    RegisteredWorkflow,
    Runtime,
    WorkflowRuntime,
)
from workflows.runtime.types.step_function import StepWorkerFunction
from workflows.workflow import Workflow

if TYPE_CHECKING:
    from workflows.context.context import Context


class WorkflowRuntimeRegistry:
    """
    Ensures that runtimes register each workflow once and only once for each runtime type.
    """

    def __init__(self) -> None:
        # Map each workflow instance to its runtime registrations.
        # Weakly references workflow keys so entries are GC'd when workflows are.
        self.workflows: IdentityWeakKeyDict[
            Workflow, dict[type[Runtime], RegisteredWorkflow]
        ] = IdentityWeakKeyDict()
        self.lock = Lock()
        self.run_contexts: dict[str, RegisteredRunContext] = {}

    def get_registered_workflow(
        self,
        workflow: Workflow,
        runtime: Runtime,
        workflow_function: ControlLoopFunction,
        steps: dict[str, StepWorkerFunction],
    ) -> RegisteredWorkflow:
        runtime_type = type(runtime)

        # Fast path without lock
        runtime_map = self.workflows.get(workflow)
        if runtime_map is not None and runtime_type in runtime_map:
            return runtime_map[runtime_type]
        with self.lock:
            # Double-check after acquiring lock
            runtime_map = self.workflows.get(workflow)
            if runtime_map is not None and runtime_type in runtime_map:
                return runtime_map[runtime_type]

            registered_workflow = runtime.register(workflow, workflow_function, steps)
            if registered_workflow is None:
                registered_workflow = RegisteredWorkflow(workflow_function, steps)
            if runtime_map is None:
                runtime_map = {}
                self.workflows[workflow] = runtime_map
            runtime_map[runtime_type] = registered_workflow
            return registered_workflow

    def register_run(
        self,
        run_id: str,
        workflow: Workflow,
        runtime: WorkflowRuntime,
        context: "Context",
        steps: dict[str, StepWorkerFunction],
    ) -> None:
        self.run_contexts[run_id] = RegisteredRunContext(
            run_id=run_id,
            workflow=workflow,
            runtime=runtime,
            context=context,
            steps=steps,
        )

    def get_run(self, run_id: str) -> Optional["RegisteredRunContext"]:
        return self.run_contexts.get(run_id)

    def delete_run(self, run_id: str) -> None:
        self.run_contexts.pop(run_id, None)


workflow_registry = WorkflowRuntimeRegistry()


@dataclass
class RegisteredRunContext:
    run_id: str
    workflow: Workflow
    runtime: WorkflowRuntime
    context: "Context"
    steps: dict[str, StepWorkerFunction]
