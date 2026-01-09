from typing import Annotated

import pytest
from workflows.decorators import step
from workflows.events import Event, StartEvent, StopEvent
from workflows.protocol import (
    WorkflowGraphEdge,
    WorkflowGraphNode,
    WorkflowGraphNodeEdges,
)
from workflows.representation_utils import extract_workflow_structure
from workflows.resource import Resource
from workflows.workflow import Workflow

from .conftest import DummyWorkflow  # type: ignore[import]


@pytest.fixture()
def ground_truth_repr() -> WorkflowGraphNodeEdges:
    return WorkflowGraphNodeEdges(
        nodes=[
            WorkflowGraphNode(
                id="end_step",
                label="end_step",
                node_type="step",
                title=None,
            ),
            WorkflowGraphNode(
                id="LastEvent",
                label="LastEvent",
                node_type="event",
                title=None,
                event_type="LastEvent",
                event_types=["LastEvent"],
            ),
            WorkflowGraphNode(
                id="StopEvent",
                label="StopEvent",
                node_type="event",
                title=None,
                event_type="StopEvent",
                event_types=["StopEvent"],
            ),
            WorkflowGraphNode(
                id="middle_step",
                label="middle_step",
                node_type="step",
                title=None,
            ),
            WorkflowGraphNode(
                id="OneTestEvent",
                label="OneTestEvent",
                node_type="event",
                title=None,
                event_type="OneTestEvent",
                event_types=["OneTestEvent"],
            ),
            WorkflowGraphNode(
                id="start_step",
                label="start_step",
                node_type="step",
                title=None,
            ),
            WorkflowGraphNode(
                id="StartEvent",
                label="StartEvent",
                node_type="event",
                title=None,
                event_type="StartEvent",
                event_types=["StartEvent"],
            ),
        ],
        edges=[
            WorkflowGraphEdge(source="end_step", target="StopEvent"),
            WorkflowGraphEdge(source="LastEvent", target="end_step"),
            WorkflowGraphEdge(source="middle_step", target="LastEvent"),
            WorkflowGraphEdge(source="OneTestEvent", target="middle_step"),
            WorkflowGraphEdge(source="start_step", target="OneTestEvent"),
            WorkflowGraphEdge(source="StartEvent", target="start_step"),
        ],
    )


def test_extract_workflow_structure(ground_truth_repr: WorkflowGraphNodeEdges) -> None:
    wf = DummyWorkflow()
    graph = extract_workflow_structure(workflow=wf)
    assert isinstance(graph, WorkflowGraphNodeEdges)
    assert sorted(
        [node.id for node in ground_truth_repr.nodes if node.node_type == "step"]
    ) == sorted([node.id for node in graph.nodes if node.node_type == "step"])
    assert sorted(
        [node.id for node in ground_truth_repr.nodes if node.node_type == "event"]
    ) == sorted([node.id for node in graph.nodes if node.node_type == "event"])
    expected_edges = ground_truth_repr.edges
    for edge in expected_edges:
        assert edge in graph.edges


def test_extract_workflow_structure_trim_label() -> None:
    wf = DummyWorkflow()
    graph = extract_workflow_structure(workflow=wf, max_label_length=2)
    assert sorted(["e*", "m*", "s*"]) == sorted(
        [node.label for node in graph.nodes if node.node_type == "step"]
    )
    assert sorted(["S*", "S*", "O*", "L*"]) == sorted(
        [node.label for node in graph.nodes if node.node_type == "event"]
    )


def test_graph_serialization() -> None:
    """Test that WorkflowGraphNodeEdges serializes correctly to JSON."""
    graph = WorkflowGraphNodeEdges(
        nodes=[
            WorkflowGraphNode(id="test", label="test", node_type="step", title=None),
            WorkflowGraphNode(
                id="OneTestEvent",
                label="OneTestEvent",
                node_type="event",
                title=None,
                event_type="OneTestEvent",
                event_types=["OneTestEvent"],
            ),
        ],
        edges=[WorkflowGraphEdge(source="test", target="OneTestEvent")],
    )
    # Test direct access
    assert len(graph.nodes) == 2
    assert graph.nodes[0].event_type is None
    assert graph.nodes[0].title is None
    assert graph.nodes[0].node_type == "step"
    assert graph.nodes[0].label == "test"
    assert graph.nodes[0].id == "test"
    assert graph.nodes[1].event_type == "OneTestEvent"
    assert graph.nodes[1].event_types == ["OneTestEvent"]
    assert graph.nodes[1].title is None
    assert graph.nodes[1].node_type == "event"
    assert graph.nodes[1].label == "OneTestEvent"
    assert graph.nodes[1].id == "OneTestEvent"
    assert len(graph.edges) == 1
    assert graph.edges[0].source == "test"
    assert graph.edges[0].target == "OneTestEvent"

    # Test JSON serialization (round-trip works)
    data = graph.model_dump()
    assert data["nodes"][0]["event_type"] is None
    assert data["nodes"][1]["event_type"] == "OneTestEvent"
    assert data["nodes"][1]["event_types"] == ["OneTestEvent"]

    # Test deserialization
    restored = WorkflowGraphNodeEdges.model_validate(data)
    assert restored.nodes[1].event_type == "OneTestEvent"
    assert restored.nodes[1].is_subclass_of("OneTestEvent")


# --- Resource node tests ---


class DatabaseClient:
    """A mock database client for testing resources."""

    pass


def get_database_client() -> DatabaseClient:
    """Factory function to create a database client.

    This docstring should appear in the resource metadata.
    """
    return DatabaseClient()


class MiddleEvent(Event):
    pass


class WorkflowWithResources(Workflow):
    @step
    async def start_step(self, ev: StartEvent) -> MiddleEvent:
        return MiddleEvent()

    @step
    async def step_with_resource(
        self,
        ev: MiddleEvent,
        db_client: Annotated[DatabaseClient, Resource(get_database_client)],
    ) -> StopEvent:
        return StopEvent(result="done")


def test_extract_workflow_structure_with_resources() -> None:
    """Test that resource nodes are extracted from workflow with resources."""
    wf = WorkflowWithResources()
    graph = extract_workflow_structure(workflow=wf)

    # Should have resource nodes
    resource_nodes = [n for n in graph.nodes if n.node_type == "resource"]
    assert len(resource_nodes) == 1

    resource_node = resource_nodes[0]
    assert resource_node.node_type == "resource"
    assert resource_node.type_name == "DatabaseClient"
    assert resource_node.getter_name == "get_database_client"
    assert resource_node.docstring is not None
    assert "Factory function" in resource_node.docstring
    assert resource_node.source_file is not None
    assert resource_node.source_line is not None
    assert resource_node.unique_hash is not None


def test_resource_node_edges_have_variable_names() -> None:
    """Test that edges from steps to resources have the variable name as label."""
    wf = WorkflowWithResources()
    graph = extract_workflow_structure(workflow=wf)

    # Find edges to resource nodes
    resource_edges = [e for e in graph.edges if e.target.startswith("resource_")]

    assert len(resource_edges) == 1
    edge = resource_edges[0]
    assert edge.label == "db_client"  # The variable name
    assert edge.source == "step_with_resource"


def test_resource_nodes_are_deduplicated() -> None:
    """Test that the same resource used in multiple steps appears only once."""

    class StepEvent(Event):
        pass

    class WorkflowWithSharedResource(Workflow):
        @step
        async def start_step(self, ev: StartEvent) -> StepEvent:
            return StepEvent()

        @step
        async def step_one(
            self,
            ev: StepEvent,
            db: Annotated[DatabaseClient, Resource(get_database_client)],
        ) -> MiddleEvent:
            return MiddleEvent()

        @step
        async def step_two(
            self,
            ev: MiddleEvent,
            db: Annotated[DatabaseClient, Resource(get_database_client)],
        ) -> StopEvent:
            return StopEvent(result="done")

    wf = WorkflowWithSharedResource()
    graph = extract_workflow_structure(workflow=wf)

    # Should have only one resource node (deduplicated)
    resource_nodes = [n for n in graph.nodes if n.node_type == "resource"]
    assert len(resource_nodes) == 1

    # But should have two edges (one from each step)
    resource_edges = [e for e in graph.edges if e.target.startswith("resource_")]
    assert len(resource_edges) == 2

    # Both edges should have the variable name "db"
    for edge in resource_edges:
        assert edge.label == "db"


def test_multiple_different_resources() -> None:
    """Test workflow with multiple different resources."""

    class CacheClient:
        pass

    def get_cache_client() -> CacheClient:
        return CacheClient()

    class WorkflowWithMultipleResources(Workflow):
        @step
        async def start_step(
            self,
            ev: StartEvent,
            db: Annotated[DatabaseClient, Resource(get_database_client)],
            cache: Annotated[CacheClient, Resource(get_cache_client)],
        ) -> StopEvent:
            return StopEvent(result="done")

    wf = WorkflowWithMultipleResources()
    graph = extract_workflow_structure(workflow=wf)

    # Should have two different resource nodes
    resource_nodes = [n for n in graph.nodes if n.node_type == "resource"]
    assert len(resource_nodes) == 2

    type_names = {rn.type_name for rn in resource_nodes}
    assert type_names == {"DatabaseClient", "CacheClient"}

    # Should have two edges with different labels
    resource_edges = [e for e in graph.edges if e.target.startswith("resource_")]
    assert len(resource_edges) == 2

    labels = {e.label for e in resource_edges}
    assert labels == {"db", "cache"}


def test_resource_node_serialization() -> None:
    """Test that WorkflowGraphNode resource serializes correctly."""
    resource_node = WorkflowGraphNode(
        id="resource_abc123",
        label="TestType",
        node_type="resource",
        type_name="TestType",
        getter_name="get_test_type",
        source_file="/path/to/file.py",
        source_line=42,
        docstring="Test docstring",
        unique_hash="abc123",
    )

    assert resource_node.id == "resource_abc123"
    assert resource_node.label == "TestType"
    assert resource_node.node_type == "resource"
    assert resource_node.type_name == "TestType"
    assert resource_node.getter_name == "get_test_type"
    assert resource_node.source_file == "/path/to/file.py"
    assert resource_node.source_line == 42
    assert resource_node.docstring == "Test docstring"
    assert resource_node.unique_hash == "abc123"

    # Test serialization
    data = resource_node.model_dump()
    assert data["id"] == "resource_abc123"
    assert data["type_name"] == "TestType"


def test_graph_with_resources() -> None:
    """Test that workflow graph with resources is correct."""
    wf = WorkflowWithResources()
    graph = extract_workflow_structure(workflow=wf)

    # Check resource nodes are in the nodes list
    resource_nodes = [n for n in graph.nodes if n.node_type == "resource"]
    assert len(resource_nodes) == 1
    rn = resource_nodes[0]
    assert rn.type_name == "DatabaseClient"
    assert rn.getter_name == "get_database_client"

    # Check edges with labels
    resource_edges = [e for e in graph.edges if e.label is not None]
    assert len(resource_edges) == 1
    assert resource_edges[0].label == "db_client"


def test_edge_with_label() -> None:
    """Test that WorkflowGraphEdge with label works correctly."""
    edge = WorkflowGraphEdge(source="resource_123", target="my_step", label="my_var")

    assert edge.source == "resource_123"
    assert edge.target == "my_step"
    assert edge.label == "my_var"


def test_edge_without_label() -> None:
    """Test that WorkflowGraphEdge without label works correctly."""
    edge = WorkflowGraphEdge(source="event_A", target="step_B")

    assert edge.source == "event_A"
    assert edge.target == "step_B"
    assert edge.label is None
