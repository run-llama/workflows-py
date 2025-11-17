from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Optional

from workflows.runtime.types._identity_weak_ref import IdentityWeakKeyDict
from workflows.runtime.types.plugin import (
    ControlLoopFunction,
    Plugin,
    RegisteredWorkflow,
    WorkflowRuntime,
)
from workflows.runtime.types.step_function import StepWorkerFunction
from workflows.workflow import Workflow

if TYPE_CHECKING:
    from workflows.context.context import Context


class WorkflowPluginRegistry:
    """
    Ensures that plugins register each workflow once and only once for each plugin.
    """

    def __init__(self) -> None:
        # Map each workflow instance to its plugin registrations.
        # Weakly references workflow keys so entries are GC'd when workflows are.
        self.workflows: IdentityWeakKeyDict[
            Workflow, dict[type[Plugin], RegisteredWorkflow]
        ] = IdentityWeakKeyDict()
        self.lock = Lock()
        self.run_contexts: dict[str, RegisteredRunContext] = {}

    def get_registered_workflow(
        self,
        workflow: Workflow,
        plugin: Plugin,
        workflow_function: ControlLoopFunction,
        steps: dict[str, StepWorkerFunction],
    ) -> RegisteredWorkflow:
        plugin_type = type(plugin)

        # Fast path without lock
        plugin_map = self.workflows.get(workflow)
        if plugin_map is not None and plugin_type in plugin_map:
            return plugin_map[plugin_type]
        with self.lock:
            # Double-check after acquiring lock
            plugin_map = self.workflows.get(workflow)
            if plugin_map is not None and plugin_type in plugin_map:
                return plugin_map[plugin_type]

            registered_workflow = plugin.register(workflow, workflow_function, steps)
            if registered_workflow is None:
                registered_workflow = RegisteredWorkflow(workflow_function, steps)
            if plugin_map is None:
                plugin_map = {}
                self.workflows[workflow] = plugin_map
            plugin_map[plugin_type] = registered_workflow
            return registered_workflow

    def register_run(
        self,
        run_id: str,
        workflow: Workflow,
        plugin: WorkflowRuntime,
        context: "Context",
        steps: dict[str, StepWorkerFunction],
    ) -> None:
        self.run_contexts[run_id] = RegisteredRunContext(
            run_id=run_id,
            workflow=workflow,
            plugin=plugin,
            context=context,
            steps=steps,
        )

    def get_run(self, run_id: str) -> Optional["RegisteredRunContext"]:
        return self.run_contexts.get(run_id)

    def delete_run(self, run_id: str) -> None:
        self.run_contexts.pop(run_id, None)


workflow_registry = WorkflowPluginRegistry()


@dataclass
class RegisteredRunContext:
    run_id: str
    workflow: Workflow
    plugin: WorkflowRuntime
    context: "Context"
    steps: dict[str, StepWorkerFunction]
