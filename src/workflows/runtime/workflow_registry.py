from threading import Lock
from weakref import WeakKeyDictionary
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
        # Map each workflow instance to its plugin registrations.
        # Weakly references workflow keys so entries are GC'd when workflows are.
        self.workflows: WeakKeyDictionary[
            Workflow, dict[type[Plugin], RegisteredWorkflow]
        ] = WeakKeyDictionary()
        self.lock = Lock()

    def get_registered_workflow(
        self,
        workflow: Workflow,
        plugin: Plugin,
        workflow_function: ControlLoopFunction,
        steps: dict[str, StepWorkerFunction[R]],
    ) -> RegisteredWorkflow:
        return RegisteredWorkflow(workflow_function, steps)
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


workflow_registry = WorkflowPluginRegistry()
