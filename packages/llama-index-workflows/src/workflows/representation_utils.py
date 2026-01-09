import hashlib
import inspect
from typing import Optional

from workflows import Workflow
from workflows.decorators import StepConfig, StepFunction
from workflows.events import (
    Event,
    HumanResponseEvent,
    InputRequiredEvent,
    StopEvent,
)
from workflows.protocol import (
    WorkflowGraphEdge,
    WorkflowGraphNode,
    WorkflowGraphNodeEdges,
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


def _create_resource_node(resource_def: ResourceDefinition) -> WorkflowGraphNode:
    """Create a WorkflowGraphNode from a ResourceDefinition.

    Extracts metadata (source file, line number, docstring) lazily here
    rather than at Resource creation time for performance.
    """
    resource = resource_def.resource
    factory = resource._factory

    # Get type name from annotation
    type_name: Optional[str] = None
    if resource_def.type_annotation is not None:
        type_annotation = resource_def.type_annotation
        if hasattr(type_annotation, "__name__"):
            type_name = type_annotation.__name__
        else:
            type_name = str(type_annotation)

    # Extract source metadata lazily
    source_file: Optional[str] = None
    source_line: Optional[int] = None
    try:
        source_file = inspect.getfile(factory)
    except (TypeError, OSError):
        pass
    try:
        _, source_line = inspect.getsourcelines(factory)
    except (TypeError, OSError):
        pass
    docstring = inspect.getdoc(factory)

    # Compute unique hash for deduplication
    hash_input = f"{resource.name}:{source_file or 'unknown'}"
    unique_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:12]

    return WorkflowGraphNode(
        id=f"resource_{unique_hash}",
        label=type_name or resource.name,
        node_type="resource",
        type_name=type_name,
        getter_name=resource.name,
        source_file=source_file,
        source_line=source_line,
        docstring=docstring,
        unique_hash=unique_hash,
    )


def _truncate_label(label: str, max_length: int) -> str:
    """Helper to truncate long labels."""
    return label if len(label) <= max_length else f"{label[: max_length - 1]}*"


def extract_workflow_structure(
    workflow: Workflow, max_label_length: Optional[int] = None
) -> WorkflowGraphNodeEdges:
    """Extract workflow structure into a graph representation."""
    # Get workflow steps
    steps: dict[str, StepFunction] = get_steps_from_class(workflow)
    if not steps:
        steps = get_steps_from_instance(workflow)

    nodes: list[WorkflowGraphNode] = []
    edges: list[WorkflowGraphEdge] = []
    added_nodes: set[str] = set()  # Track added node IDs to avoid duplicates
    added_resource_nodes: dict[int, WorkflowGraphNode] = {}  # Track by factory id

    step_config: Optional[StepConfig] = None

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
        step_label = (
            _truncate_label(step_name, max_label_length)
            if max_label_length
            else step_name
        )
        step_title = (
            step_name
            if max_label_length and len(step_name) > max_label_length
            else None
        )

        if step_name not in added_nodes:
            nodes.append(
                WorkflowGraphNode(
                    id=step_name,
                    label=step_label,
                    node_type="step",
                    title=step_title,
                )
            )
            added_nodes.add(step_name)

        # Add event nodes for accepted events
        for event_type in step_config.accepted_events:
            if event_type == StopEvent and event_type != current_stop_event:
                continue

            event_label = (
                _truncate_label(event_type.__name__, max_label_length)
                if max_label_length
                else event_type.__name__
            )
            event_title = (
                event_type.__name__
                if max_label_length and len(event_type.__name__) > max_label_length
                else None
            )

            if event_type.__name__ not in added_nodes:
                nodes.append(
                    WorkflowGraphNode(
                        id=event_type.__name__,
                        label=event_label,
                        node_type="event",
                        title=event_title,
                        event_type=event_type.__name__,
                        event_types=_get_event_type_chain(event_type),
                    )
                )
                added_nodes.add(event_type.__name__)

        # Add event nodes for return types
        for return_type in step_config.return_types:
            if return_type is type(None):
                continue

            return_label = (
                _truncate_label(return_type.__name__, max_label_length)
                if max_label_length
                else return_type.__name__
            )
            return_title = (
                return_type.__name__
                if max_label_length and len(return_type.__name__) > max_label_length
                else None
            )

            if return_type.__name__ not in added_nodes:
                nodes.append(
                    WorkflowGraphNode(
                        id=return_type.__name__,
                        label=return_label,
                        node_type="event",
                        title=return_title,
                        event_type=return_type.__name__,
                        event_types=_get_event_type_chain(return_type),
                    )
                )
                added_nodes.add(return_type.__name__)

            # Add external_step node when InputRequiredEvent is found
            if (
                issubclass(return_type, InputRequiredEvent)
                and "external_step" not in added_nodes
            ):
                nodes.append(
                    WorkflowGraphNode(
                        id="external_step",
                        label="external_step",
                        node_type="external",
                    )
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

    return WorkflowGraphNodeEdges(nodes=nodes, edges=edges)
