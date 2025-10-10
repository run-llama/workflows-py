from dataclasses import dataclass
from typing import List, Optional

from workflows.events import (
    StopEvent,
    InputRequiredEvent,
    HumanResponseEvent,
)
from workflows.decorators import StepConfig
from workflows.protocol import (
    WorkflowGraphEdge,
    WorkflowGraphNode,
    WorkflowGraphNodeEdges,
)
from workflows.utils import (
    get_steps_from_class,
    get_steps_from_instance,
)
from workflows import Workflow


@dataclass
class DrawWorkflowNode:
    """Represents a node in the workflow graph."""

    id: str
    label: str
    node_type: str  # 'step', 'event', 'external'
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
class DrawWorkflowEdge:
    """Represents an edge in the workflow graph."""

    source: str
    target: str

    def to_response_model(self) -> WorkflowGraphEdge:
        return WorkflowGraphEdge(
            source=self.source,
            target=self.target,
        )


@dataclass
class DrawWorkflowGraph:
    """Intermediate representation of workflow structure."""

    nodes: List[DrawWorkflowNode]
    edges: List[DrawWorkflowEdge]

    def to_response_model(self) -> WorkflowGraphNodeEdges:
        return WorkflowGraphNodeEdges(
            nodes=[node.to_response_model() for node in self.nodes],
            edges=[edge.to_response_model() for edge in self.edges],
        )


def _truncate_label(label: str, max_length: int) -> str:
    """Helper to truncate long labels."""
    return label if len(label) <= max_length else f"{label[: max_length - 1]}*"


def _extract_workflow_structure(
    workflow: Workflow, max_label_length: Optional[int] = None
) -> DrawWorkflowGraph:
    """Extract workflow structure into an intermediate representation."""
    # Get workflow steps
    steps = get_steps_from_class(workflow)
    if not steps:
        steps = get_steps_from_instance(workflow)

    nodes = []
    edges = []
    added_nodes = set()  # Track added node IDs to avoid duplicates

    step_config: Optional[StepConfig] = None

    # Only one kind of `StopEvent` is allowed in a `Workflow`.
    # Assuming that `Workflow` is validated before drawing, it's enough to find the first one.
    current_stop_event = None
    for step_name, step_func in steps.items():
        step_config = getattr(step_func, "__step_config", None)
        if step_config is None:
            continue

        for return_type in step_config.return_types:
            if issubclass(return_type, StopEvent):
                current_stop_event = return_type
                break

        if current_stop_event:
            break

    # First pass: Add all nodes
    for step_name, step_func in steps.items():
        step_config = getattr(step_func, "__step_config", None)
        if step_config is None:
            continue

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

    # Second pass: Add edges
    for step_name, step_func in steps.items():
        step_config = getattr(step_func, "__step_config", None)
        if step_config is None:
            continue

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

    return DrawWorkflowGraph(nodes=nodes, edges=edges)
