from __future__ import annotations

import hashlib
import inspect

from workflows import Workflow
from workflows.decorators import StepConfig, StepFunction
from workflows.events import (
    Event,
    HumanResponseEvent,
    InputRequiredEvent,
    StopEvent,
)
from workflows.representation.types import (
    WorkflowEventNode,
    WorkflowExternalNode,
    WorkflowGraph,
    WorkflowGraphEdge,
    WorkflowGraphNode,
    WorkflowResourceNode,
    WorkflowStepNode,
)
from workflows.resource import ResourceDefinition
from workflows.utils import (
    get_steps_from_class,
    get_steps_from_instance,
)


def _get_event_type_chain(cls: type) -> list[str]:
    """Get the event type inheritance chain including the class itself.

    Returns a list starting with the class name, followed by parent Event
    subclasses up to (but not including) Event itself.
    """
    names: list[str] = [cls.__name__]
    for parent in cls.mro()[1:]:
        if parent is Event:
            break
        if isinstance(parent, type) and issubclass(parent, Event):
            names.append(parent.__name__)
    return names


def _create_resource_node(resource_def: ResourceDefinition) -> WorkflowResourceNode:
    """Create a WorkflowResourceNode from a ResourceDefinition.

    Extracts metadata (source file, line number, docstring) lazily here
    rather than at Resource creation time for performance.
    """
    resource = resource_def.resource
    factory = resource._factory

    # Get type name from annotation
    type_name: str | None = None
    if resource_def.type_annotation is not None:
        type_annotation = resource_def.type_annotation
        if hasattr(type_annotation, "__name__"):
            type_name = type_annotation.__name__
        else:
            type_name = str(type_annotation)

    # Extract source metadata lazily
    source_file: str | None = None
    source_line: int | None = None
    try:
        source_file = inspect.getfile(factory)
    except (TypeError, OSError):
        pass
    try:
        _, source_line = inspect.getsourcelines(factory)
    except (TypeError, OSError):
        pass
    resource_description = inspect.getdoc(factory)

    # Compute unique hash for deduplication
    hash_input = f"{resource.name}:{source_file or 'unknown'}"
    unique_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:12]

    # Label: prefer type_name, then getter_name, then id
    node_id = f"resource_{unique_hash}"
    label = type_name or resource.name or node_id

    return WorkflowResourceNode(
        id=node_id,
        label=label,
        type_name=type_name,
        getter_name=resource.name,
        source_file=source_file,
        source_line=source_line,
        description=resource_description,
    )


def get_workflow_representation(workflow: Workflow) -> WorkflowGraph:
    """Build a graph representation of a workflow's structure.

    Extracts the workflow's steps, events, and resources into a WorkflowGraph
    that can be used for visualization or analysis.

    Args:
        workflow: The workflow instance to build a representation for.

    Returns:
        A WorkflowGraph containing nodes for steps, events, resources,
        and external interactions, with edges showing the data flow.
    """
    # Get workflow steps
    steps: dict[str, StepFunction] = get_steps_from_class(workflow)
    if not steps:
        steps = get_steps_from_instance(workflow)

    nodes: list[WorkflowGraphNode] = []
    edges: list[WorkflowGraphEdge] = []
    added_nodes: set[str] = set()  # Track added node IDs to avoid duplicates
    added_resource_nodes: dict[int, WorkflowResourceNode] = {}  # Track by factory id

    step_config: StepConfig | None = None

    # Only one kind of `StopEvent` is allowed in a `Workflow`.
    # Assuming that `Workflow` is validated before drawing, it's enough to find the first one.
    current_stop_event = None
    for step_name, step_func in steps.items():
        step_config = step_func._step_config

        for return_type in step_config.return_types:
            if issubclass(return_type, StopEvent):
                current_stop_event = return_type
                break

        if current_stop_event:
            break

    # First pass: Add all nodes
    for step_name, step_func in steps.items():
        step_config = step_func._step_config

        # Add step node
        if step_name not in added_nodes:
            step_description = inspect.getdoc(step_func)
            nodes.append(
                WorkflowStepNode(
                    id=step_name, label=step_name, description=step_description
                )
            )
            added_nodes.add(step_name)

        # Add event nodes for accepted events
        for event_type in step_config.accepted_events:
            if event_type == StopEvent and event_type != current_stop_event:
                continue

            if event_type.__name__ not in added_nodes:
                nodes.append(
                    WorkflowEventNode(
                        id=event_type.__name__,
                        label=event_type.__name__,
                        event_type=event_type.__name__,
                        event_types=_get_event_type_chain(event_type),
                        event_schema=event_type.model_json_schema(),
                    )
                )
                added_nodes.add(event_type.__name__)

        # Add event nodes for return types
        for return_type in step_config.return_types:
            if return_type is type(None):
                continue

            if return_type.__name__ not in added_nodes:
                nodes.append(
                    WorkflowEventNode(
                        id=return_type.__name__,
                        label=return_type.__name__,
                        event_type=return_type.__name__,
                        event_types=_get_event_type_chain(return_type),
                        event_schema=return_type.model_json_schema(),
                    )
                )
                added_nodes.add(return_type.__name__)

            # Add external_step node when InputRequiredEvent is found
            if (
                issubclass(return_type, InputRequiredEvent)
                and "external_step" not in added_nodes
            ):
                nodes.append(
                    WorkflowExternalNode(id="external_step", label="external_step")
                )
                added_nodes.add("external_step")

        # Add resource nodes (deduplicated by factory identity)
        for resource_def in step_config.resources:
            factory_id = id(resource_def.resource._factory)
            if factory_id not in added_resource_nodes:
                resource_node = _create_resource_node(resource_def)
                nodes.append(resource_node)
                added_resource_nodes[factory_id] = resource_node

    # Second pass: Add edges
    for step_name, step_func in steps.items():
        step_config = step_func._step_config

        # Edges from steps to return types
        for return_type in step_config.return_types:
            if return_type is not type(None):
                edges.append(
                    WorkflowGraphEdge(source=step_name, target=return_type.__name__)
                )

            if issubclass(return_type, InputRequiredEvent):
                edges.append(
                    WorkflowGraphEdge(
                        source=return_type.__name__, target="external_step"
                    )
                )

        # Edges from events to steps
        for event_type in step_config.accepted_events:
            if step_name == "_done" and issubclass(event_type, StopEvent):
                if current_stop_event:
                    edges.append(
                        WorkflowGraphEdge(
                            source=current_stop_event.__name__, target=step_name
                        )
                    )
            else:
                edges.append(
                    WorkflowGraphEdge(source=event_type.__name__, target=step_name)
                )

            if issubclass(event_type, HumanResponseEvent):
                edges.append(
                    WorkflowGraphEdge(
                        source="external_step", target=event_type.__name__
                    )
                )

        # Edges from steps to resources (with variable name as label)
        for resource_def in step_config.resources:
            factory_id = id(resource_def.resource._factory)
            resource_node = added_resource_nodes[factory_id]
            edges.append(
                WorkflowGraphEdge(
                    source=step_name,
                    target=resource_node.id,
                    label=resource_def.name,  # The variable name
                )
            )

    workflow_name = type(workflow).__name__
    workflow_description = inspect.getdoc(workflow)
    return WorkflowGraph(
        name=workflow_name, nodes=nodes, edges=edges, description=workflow_description
    )


__all__ = ["get_workflow_representation"]
