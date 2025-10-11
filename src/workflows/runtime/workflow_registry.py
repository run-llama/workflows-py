from threading import Lock
from typing import Any
from workflows.runtime.types.plugin import (
    ControlLoopFunction,
    Plugin,
    RegisteredWorkflow,
)
from workflows.workflow import Workflow
from workflows.runtime.types.step_function import StepWorkerFunction
from workflows.decorators import R


class WorkflowPluginRegistry:
    """
    Ensures that plugins register each workflow once and only once for each plugin.
    """

    def __init__(self) -> None:
        self.workflows: dict[tuple[Any, type[Plugin]], RegisteredWorkflow] = {}
        self.lock = Lock()

    def get_registered_workflow(
        self,
        workflow: Workflow,
        plugin: Plugin,
        workflow_function: ControlLoopFunction,
        steps: dict[str, StepWorkerFunction[R]],
    ) -> RegisteredWorkflow:
        workflow_id = id(
            workflow
        )  # type(workflow) - consider making this configurable? There's some weird scenarios where different workflow instances are used as stateful things across runs (some tests use the same workflow instance for multiple runs)
        plugin_type = type(plugin)
        if (workflow_id, plugin_type) in self.workflows:
            return self.workflows[(workflow_id, plugin_type)]
        with self.lock:
            if (workflow_id, plugin_type) in self.workflows:
                return self.workflows[(workflow_id, plugin_type)]

            registered_workflow = plugin.register(workflow, workflow_function, steps)
            if registered_workflow is None:
                registered_workflow = RegisteredWorkflow(workflow_function, steps)
            self.workflows[(workflow_id, plugin_type)] = registered_workflow
            return registered_workflow


workflow_registry = WorkflowPluginRegistry()
