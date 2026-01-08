from dataclasses import dataclass, field
from typing import List, Optional

from workflows import Workflow
from workflows.decorators import StepConfig, StepFunction
from workflows.events import (
    HumanResponseEvent,
    InputRequiredEvent,
    StopEvent,
)
from workflows.protocol import (
    WorkflowGraphEdge,
    WorkflowGraphNode,
    WorkflowGraphNodeEdges,
    WorkflowGraphResourceNode,
)
from workflows.resource import ResourceDefinition
from workflows.utils import (
    get_steps_from_class,
    get_steps_from_instance,
)


@dataclass
class DrawWorkflowNode:
    """Represents a node in the workflow graph."""

    id: str
    label: str
    node_type: str  # 'step', 'event', 'external', 'resource'
    title: Optional[str] = None
    event_type: Optional[type] = (
        None  # Store the actual event type for styling decisions
    )

    def to_response_model(self) -> WorkflowGraphNode:
        return WorkflowGraphNode(
            id=self.id,
            label=self.label,
            node_type=self.node_type,
            title=self.title,
            event_type=self.event_type.__name__ if self.event_type else None,
        )


@dataclass
class DrawWorkflowResourceNode:
    """Represents a resource node in the workflow graph."""

    id: str
    label: str
    node_type: str = "resource"
    type_name: Optional[str] = None  # e.g., "AsyncLlamaCloud"
    getter_name: Optional[str] = None  # e.g., "get_llama_cloud_client"
    source_file: Optional[str] = None
    source_line: Optional[int] = None
    docstring: Optional[str] = None
    unique_hash: Optional[str] = None

    def to_response_model(self) -> WorkflowGraphResourceNode:
        return WorkflowGraphResourceNode(
            id=self.id,
            label=self.label,
            node_type=self.node_type,
            type_name=self.type_name,
            getter_name=self.getter_name,
            source_file=self.source_file,
            source_line=self.source_line,
            docstring=self.docstring,
            unique_hash=self.unique_hash,
        )

    @classmethod
    def from_resource_definition(
        cls, resource_def: ResourceDefinition
    ) -> "DrawWorkflowResourceNode":
        """Create a DrawWorkflowResourceNode from a ResourceDefinition."""
        resource = resource_def.resource

        # Get type name from annotation
        type_name: Optional[str] = None
        if resource_def.type_annotation is not None:
            type_annotation = resource_def.type_annotation
            if hasattr(type_annotation, "__name__"):
                type_name = type_annotation.__name__
            else:
                type_name = str(type_annotation)

        return cls(
            id=f"resource_{resource.unique_id}",
            label=type_name or resource.name,
            node_type="resource",
            type_name=type_name,
            getter_name=resource.name,
            source_file=resource.source_file,
            source_line=resource.source_line,
            docstring=resource.docstring,
            unique_hash=resource.unique_id,
        )


@dataclass
class DrawWorkflowEdge:
    """Represents an edge in the workflow graph."""

    source: str
    target: str
    label: Optional[str] = None  # Edge label (e.g., variable name for resources)

    def to_response_model(self) -> WorkflowGraphEdge:
        return WorkflowGraphEdge(
            source=self.source,
            target=self.target,
            label=self.label,
        )


@dataclass
class DrawWorkflowGraph:
    """Intermediate representation of workflow structure."""

    nodes: List[DrawWorkflowNode]
    edges: List[DrawWorkflowEdge]
    resource_nodes: List[DrawWorkflowResourceNode] = field(default_factory=list)

    def to_response_model(self) -> WorkflowGraphNodeEdges:
        return WorkflowGraphNodeEdges(
            nodes=[node.to_response_model() for node in self.nodes],
            edges=[edge.to_response_model() for edge in self.edges],
            resource_nodes=[rn.to_response_model() for rn in self.resource_nodes],
        )


def _truncate_label(label: str, max_length: int) -> str:
    """Helper to truncate long labels."""
    return label if len(label) <= max_length else f"{label[: max_length - 1]}*"


def extract_workflow_structure(
    workflow: Workflow, max_label_length: Optional[int] = None
) -> DrawWorkflowGraph:
    """Extract workflow structure into an intermediate representation."""
    # Get workflow steps
    steps: dict[str, StepFunction] = get_steps_from_class(workflow)
    if not steps:
        steps = get_steps_from_instance(workflow)

    nodes: List[DrawWorkflowNode] = []
    edges: List[DrawWorkflowEdge] = []
    resource_nodes: List[DrawWorkflowResourceNode] = []
    added_nodes: set[str] = set()  # Track added node IDs to avoid duplicates
    added_resource_nodes: dict[
        str, DrawWorkflowResourceNode
    ] = {}  # Track by unique_hash

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
                DrawWorkflowNode(
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
                    DrawWorkflowNode(
                        id=event_type.__name__,
                        label=event_label,
                        node_type="event",
                        title=event_title,
                        event_type=event_type,
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
                    DrawWorkflowNode(
                        id=return_type.__name__,
                        label=return_label,
                        node_type="event",
                        title=return_title,
                        event_type=return_type,
                    )
                )
                added_nodes.add(return_type.__name__)

            # Add external_step node when InputRequiredEvent is found
            if (
                issubclass(return_type, InputRequiredEvent)
                and "external_step" not in added_nodes
            ):
                nodes.append(
                    DrawWorkflowNode(
                        id="external_step",
                        label="external_step",
                        node_type="external",
                    )
                )
                added_nodes.add("external_step")

        # Add resource nodes (deduplicated by unique_hash)
        for resource_def in step_config.resources:
            resource_hash = resource_def.resource.unique_id
            if resource_hash not in added_resource_nodes:
                resource_node = DrawWorkflowResourceNode.from_resource_definition(
                    resource_def
                )
                resource_nodes.append(resource_node)
                added_resource_nodes[resource_hash] = resource_node

    # Second pass: Add edges
    for step_name, step_func in steps.items():
        step_config = step_func._step_config

        # Edges from steps to return types
        for return_type in step_config.return_types:
            if return_type is not type(None):
                edges.append(DrawWorkflowEdge(step_name, return_type.__name__))

            if issubclass(return_type, InputRequiredEvent):
                edges.append(DrawWorkflowEdge(return_type.__name__, "external_step"))

        # Edges from events to steps
        for event_type in step_config.accepted_events:
            if step_name == "_done" and issubclass(event_type, StopEvent):
                if current_stop_event:
                    edges.append(
                        DrawWorkflowEdge(current_stop_event.__name__, step_name)
                    )
            else:
                edges.append(DrawWorkflowEdge(event_type.__name__, step_name))

            if issubclass(event_type, HumanResponseEvent):
                edges.append(DrawWorkflowEdge("external_step", event_type.__name__))

        # Edges from resources to steps (with variable name as label)
        for resource_def in step_config.resources:
            resource_hash = resource_def.resource.unique_id
            resource_node = added_resource_nodes[resource_hash]
            edges.append(
                DrawWorkflowEdge(
                    source=resource_node.id,
                    target=step_name,
                    label=resource_def.name,  # The variable name
                )
            )

    return DrawWorkflowGraph(nodes=nodes, edges=edges, resource_nodes=resource_nodes)
