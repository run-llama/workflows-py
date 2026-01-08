from typing import Annotated

import pytest
from workflows.decorators import step
from workflows.events import Event, StartEvent, StopEvent
from workflows.representation_utils import (
    DrawWorkflowEdge,
    DrawWorkflowGraph,
    DrawWorkflowNode,
    extract_workflow_structure,
)
from workflows.resource import Resource
from workflows.workflow import Workflow

from .conftest import DummyWorkflow, LastEvent, OneTestEvent  # type: ignore[import]


@pytest.fixture()
def ground_truth_repr() -> DrawWorkflowGraph:
    return DrawWorkflowGraph(
        nodes=[
            DrawWorkflowNode(
                id="end_step",
                label="end_step",
                node_type="step",
                title=None,
                event_type=None,
            ),
            DrawWorkflowNode(
                id="LastEvent",
                label="LastEvent",
                node_type="event",
                title=None,
                event_type=LastEvent,
            ),
            DrawWorkflowNode(
                id="StopEvent",
                label="StopEvent",
                node_type="event",
                title=None,
                event_type=StopEvent,
            ),
            DrawWorkflowNode(
                id="middle_step",
                label="middle_step",
                node_type="step",
                title=None,
                event_type=None,
            ),
            DrawWorkflowNode(
                id="OneTestEvent",
                label="OneTestEvent",
                node_type="event",
                title=None,
                event_type=OneTestEvent,
            ),
            DrawWorkflowNode(
                id="start_step",
                label="start_step",
                node_type="step",
                title=None,
                event_type=None,
            ),
            DrawWorkflowNode(
                id="StartEvent",
                label="StartEvent",
                node_type="event",
                title=None,
                event_type=StartEvent,
            ),
        ],
        edges=[
            DrawWorkflowEdge(source="end_step", target="StopEvent"),
            DrawWorkflowEdge(source="LastEvent", target="end_step"),
            DrawWorkflowEdge(source="middle_step", target="LastEvent"),
            DrawWorkflowEdge(source="OneTestEvent", target="middle_step"),
            DrawWorkflowEdge(source="start_step", target="OneTestEvent"),
            DrawWorkflowEdge(source="StartEvent", target="start_step"),
        ],
    )


def test_extract_workflow_structure(ground_truth_repr: DrawWorkflowGraph) -> None:
    wf = DummyWorkflow()
    graph = extract_workflow_structure(workflow=wf)
    assert isinstance(graph, DrawWorkflowGraph)
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


def test_graph_to_response_model() -> None:
    graph = DrawWorkflowGraph(
        nodes=[
            DrawWorkflowNode(
                id="test", label="test", node_type="step", title=None, event_type=None
            ),
            DrawWorkflowNode(
                id="OneTestEvent",
                label="OneTestEvent",
                node_type="event",
                title=None,
                event_type=OneTestEvent,
            ),
        ],
        edges=[DrawWorkflowEdge(source="test", target="OneTestEvent")],
    )
    res = graph.to_response_model()
    assert len(res.nodes) == 2
    assert res.nodes[0].event_type is None
    assert res.nodes[0].title is None
    assert res.nodes[0].node_type == "step"
    assert res.nodes[0].label == "test"
    assert res.nodes[0].id == "test"
    assert res.nodes[1].event_type == OneTestEvent.__name__
    assert res.nodes[1].title is None
    assert res.nodes[1].node_type == "event"
    assert res.nodes[1].label == "OneTestEvent"
    assert res.nodes[1].id == "OneTestEvent"
    assert len(res.edges) == 1
    assert res.edges[0].source == "test"
    assert res.edges[0].target == "OneTestEvent"


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
    assert len(graph.resource_nodes) == 1

    resource_node = graph.resource_nodes[0]
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
    assert len(graph.resource_nodes) == 1

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
    assert len(graph.resource_nodes) == 2

    type_names = {rn.type_name for rn in graph.resource_nodes}
    assert type_names == {"DatabaseClient", "CacheClient"}

    # Should have two edges with different labels
    resource_edges = [e for e in graph.edges if e.target.startswith("resource_")]
    assert len(resource_edges) == 2

    labels = {e.label for e in resource_edges}
    assert labels == {"db", "cache"}


def test_resource_node_to_response_model() -> None:
    """Test that DrawWorkflowNode converts to resource response model correctly."""
    resource_node = DrawWorkflowNode(
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

    response = resource_node.to_resource_response_model()

    assert response.id == "resource_abc123"
    assert response.label == "TestType"
    assert response.node_type == "resource"
    assert response.type_name == "TestType"
    assert response.getter_name == "get_test_type"
    assert response.source_file == "/path/to/file.py"
    assert response.source_line == 42
    assert response.docstring == "Test docstring"
    assert response.unique_hash == "abc123"


def test_graph_with_resources_to_response_model() -> None:
    """Test that DrawWorkflowGraph with resources converts correctly."""
    wf = WorkflowWithResources()
    graph = extract_workflow_structure(workflow=wf)

    response = graph.to_response_model()

    # Check resource_nodes are included
    assert len(response.resource_nodes) == 1
    rn = response.resource_nodes[0]
    assert rn.type_name == "DatabaseClient"
    assert rn.getter_name == "get_database_client"

    # Check edges with labels
    resource_edges = [e for e in response.edges if e.label is not None]
    assert len(resource_edges) == 1
    assert resource_edges[0].label == "db_client"


def test_edge_with_label_to_response_model() -> None:
    """Test that DrawWorkflowEdge with label converts correctly."""
    edge = DrawWorkflowEdge(source="resource_123", target="my_step", label="my_var")
    response = edge.to_response_model()

    assert response.source == "resource_123"
    assert response.target == "my_step"
    assert response.label == "my_var"


def test_edge_without_label_to_response_model() -> None:
    """Test that DrawWorkflowEdge without label converts correctly."""
    edge = DrawWorkflowEdge(source="event_A", target="step_B")
    response = edge.to_response_model()

    assert response.source == "event_A"
    assert response.target == "step_B"
    assert response.label is None
