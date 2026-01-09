from typing import Annotated

import pytest
from workflows.decorators import step
from workflows.events import Event, StartEvent, StopEvent
from workflows.protocol import (
    WorkflowEventNode,
    WorkflowExternalNode,
    WorkflowGraph,
    WorkflowGraphEdge,
    WorkflowResourceNode,
    WorkflowStepNode,
)
from workflows.representation_utils import extract_workflow_structure
from workflows.resource import Resource
from workflows.workflow import Workflow

from .conftest import DummyWorkflow  # type: ignore[import]


@pytest.fixture()
def ground_truth_repr() -> WorkflowGraph:
    return WorkflowGraph(
        nodes=[
            WorkflowStepNode(
                id="end_step",
                label="end_step",
            ),
            WorkflowEventNode(
                id="LastEvent",
                label="LastEvent",
                event_type="LastEvent",
                event_types=["LastEvent"],
            ),
            WorkflowEventNode(
                id="StopEvent",
                label="StopEvent",
                event_type="StopEvent",
                event_types=["StopEvent"],
            ),
            WorkflowStepNode(
                id="middle_step",
                label="middle_step",
            ),
            WorkflowEventNode(
                id="OneTestEvent",
                label="OneTestEvent",
                event_type="OneTestEvent",
                event_types=["OneTestEvent"],
            ),
            WorkflowStepNode(
                id="start_step",
                label="start_step",
            ),
            WorkflowEventNode(
                id="StartEvent",
                label="StartEvent",
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


def test_extract_workflow_structure(ground_truth_repr: WorkflowGraph) -> None:
    wf = DummyWorkflow()
    graph = extract_workflow_structure(workflow=wf)
    assert isinstance(graph, WorkflowGraph)
    assert sorted(
        [node.id for node in ground_truth_repr.nodes if node.node_type == "step"]
    ) == sorted([node.id for node in graph.nodes if node.node_type == "step"])
    assert sorted(
        [node.id for node in ground_truth_repr.nodes if node.node_type == "event"]
    ) == sorted([node.id for node in graph.nodes if node.node_type == "event"])
    expected_edges = ground_truth_repr.edges
    for edge in expected_edges:
        assert edge in graph.edges


def test_truncated_label() -> None:
    """Test that truncated_label method works correctly."""
    node = WorkflowStepNode(id="my_step", label="my_long_step_name")
    assert node.truncated_label(5) == "my_l*"
    assert node.truncated_label(20) == "my_long_step_name"
    assert node.truncated_label(17) == "my_long_step_name"


def test_graph_serialization() -> None:
    """Test that WorkflowGraphNodeEdges serializes correctly to JSON."""
    graph = WorkflowGraph(
        nodes=[
            WorkflowStepNode(id="test", label="test"),
            WorkflowEventNode(
                id="OneTestEvent",
                label="OneTestEvent",
                event_type="OneTestEvent",
                event_types=["OneTestEvent"],
            ),
        ],
        edges=[WorkflowGraphEdge(source="test", target="OneTestEvent")],
    )
    # Test direct access
    assert len(graph.nodes) == 2
    step_node = graph.nodes[0]
    assert isinstance(step_node, WorkflowStepNode)
    assert step_node.node_type == "step"
    assert step_node.label == "test"
    assert step_node.id == "test"
    event_node = graph.nodes[1]
    assert isinstance(event_node, WorkflowEventNode)
    assert event_node.event_type == "OneTestEvent"
    assert event_node.event_types == ["OneTestEvent"]
    assert event_node.node_type == "event"
    assert event_node.label == "OneTestEvent"
    assert event_node.id == "OneTestEvent"
    assert len(graph.edges) == 1
    assert graph.edges[0].source == "test"
    assert graph.edges[0].target == "OneTestEvent"

    # Test JSON serialization (round-trip works)
    data = graph.model_dump()
    assert "event_type" not in data["nodes"][0]  # Step nodes don't have event_type
    assert data["nodes"][1]["event_type"] == "OneTestEvent"
    assert data["nodes"][1]["event_types"] == ["OneTestEvent"]

    # Test deserialization
    restored = WorkflowGraph.model_validate(data)
    restored_event = restored.nodes[1]
    assert isinstance(restored_event, WorkflowEventNode)
    assert restored_event.event_type == "OneTestEvent"
    assert restored_event.is_subclass_of("OneTestEvent")


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
    resource_nodes = [n for n in graph.nodes if isinstance(n, WorkflowResourceNode)]
    assert len(resource_nodes) == 1

    resource_node = resource_nodes[0]
    assert resource_node.node_type == "resource"
    assert resource_node.type_name == "DatabaseClient"
    assert resource_node.getter_name == "get_database_client"
    assert resource_node.description is not None
    assert "Factory function" in resource_node.description
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
    resource_nodes = [n for n in graph.nodes if isinstance(n, WorkflowResourceNode)]
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
    resource_nodes = [n for n in graph.nodes if isinstance(n, WorkflowResourceNode)]
    assert len(resource_nodes) == 2

    type_names = {rn.type_name for rn in resource_nodes}
    assert type_names == {"DatabaseClient", "CacheClient"}

    # Should have two edges with different labels
    resource_edges = [e for e in graph.edges if e.target.startswith("resource_")]
    assert len(resource_edges) == 2

    labels = {e.label for e in resource_edges}
    assert labels == {"db", "cache"}


def test_resource_node_serialization() -> None:
    """Test that WorkflowResourceNode serializes correctly."""
    resource_node = WorkflowResourceNode(
        id="resource_abc123",
        label="TestType",
        type_name="TestType",
        getter_name="get_test_type",
        source_file="/path/to/file.py",
        source_line=42,
        description="Test docstring",
        unique_hash="abc123",
    )

    assert resource_node.id == "resource_abc123"
    assert resource_node.label == "TestType"
    assert resource_node.node_type == "resource"
    assert resource_node.type_name == "TestType"
    assert resource_node.getter_name == "get_test_type"
    assert resource_node.source_file == "/path/to/file.py"
    assert resource_node.source_line == 42
    assert resource_node.description == "Test docstring"
    assert resource_node.unique_hash == "abc123"

    # Test serialization
    data = resource_node.model_dump()
    assert data["id"] == "resource_abc123"
    assert data["label"] == "TestType"
    assert data["type_name"] == "TestType"
    assert data["node_type"] == "resource"

    # Test deserialization
    restored = WorkflowResourceNode.model_validate(data)
    assert restored.id == "resource_abc123"
    assert restored.label == "TestType"
    assert restored.type_name == "TestType"
    assert restored.node_type == "resource"


def test_graph_with_resources() -> None:
    """Test that workflow graph with resources is correct."""
    wf = WorkflowWithResources()
    graph = extract_workflow_structure(workflow=wf)

    # Check resource nodes are in the nodes list
    resource_nodes = [n for n in graph.nodes if isinstance(n, WorkflowResourceNode)]
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


# --- Serialization/Deserialization tests for all node types ---


def test_step_node_serialization_roundtrip() -> None:
    """Test WorkflowStepNode serialization and deserialization."""
    node = WorkflowStepNode(id="my_step", label="My Step")

    data = node.model_dump()
    assert data["id"] == "my_step"
    assert data["label"] == "My Step"
    assert data["node_type"] == "step"

    restored = WorkflowStepNode.model_validate(data)
    assert restored.id == "my_step"
    assert restored.label == "My Step"
    assert restored.node_type == "step"


def test_event_node_serialization_roundtrip() -> None:
    """Test WorkflowEventNode serialization and deserialization."""
    node = WorkflowEventNode(
        id="MyEvent",
        label="My Event",
        event_type="MyEvent",
        event_types=["MyEvent", "ParentEvent"],
    )

    data = node.model_dump()
    assert data["id"] == "MyEvent"
    assert data["label"] == "My Event"
    assert data["node_type"] == "event"
    assert data["event_type"] == "MyEvent"
    assert data["event_types"] == ["MyEvent", "ParentEvent"]

    restored = WorkflowEventNode.model_validate(data)
    assert restored.id == "MyEvent"
    assert restored.label == "My Event"
    assert restored.node_type == "event"
    assert restored.event_type == "MyEvent"
    assert restored.event_types == ["MyEvent", "ParentEvent"]
    assert restored.is_subclass_of("ParentEvent")
    assert not restored.is_subclass_of("UnrelatedEvent")


def test_external_node_serialization_roundtrip() -> None:
    """Test WorkflowExternalNode serialization and deserialization."""
    node = WorkflowExternalNode(id="external_step", label="External Step")

    data = node.model_dump()
    assert data["id"] == "external_step"
    assert data["label"] == "External Step"
    assert data["node_type"] == "external"

    restored = WorkflowExternalNode.model_validate(data)
    assert restored.id == "external_step"
    assert restored.label == "External Step"
    assert restored.node_type == "external"


def test_resource_node_serialization_roundtrip() -> None:
    """Test WorkflowResourceNode serialization and deserialization."""
    node = WorkflowResourceNode(
        id="resource_abc123",
        label="MyResourceType",
        type_name="MyResourceType",
        getter_name="get_my_resource",
        source_file="/path/to/source.py",
        source_line=100,
        description="Resource docstring",
        unique_hash="abc123",
    )

    data = node.model_dump()
    assert data["id"] == "resource_abc123"
    assert data["label"] == "MyResourceType"
    assert data["node_type"] == "resource"
    assert data["type_name"] == "MyResourceType"
    assert data["getter_name"] == "get_my_resource"
    assert data["source_file"] == "/path/to/source.py"
    assert data["source_line"] == 100
    assert data["description"] == "Resource docstring"
    assert data["unique_hash"] == "abc123"

    restored = WorkflowResourceNode.model_validate(data)
    assert restored.id == "resource_abc123"
    assert restored.label == "MyResourceType"
    assert restored.type_name == "MyResourceType"
    assert restored.getter_name == "get_my_resource"
    assert restored.node_type == "resource"


def test_graph_with_all_node_types_serialization() -> None:
    """Test full graph serialization/deserialization with all node types."""
    graph = WorkflowGraph(
        nodes=[
            WorkflowStepNode(id="step1", label="Step 1"),
            WorkflowEventNode(
                id="StartEvent",
                label="StartEvent",
                event_type="StartEvent",
                event_types=["StartEvent"],
            ),
            WorkflowExternalNode(id="external", label="External"),
            WorkflowResourceNode(
                id="resource_123",
                label="DB",
                type_name="DatabaseClient",
                getter_name="get_db",
            ),
        ],
        edges=[
            WorkflowGraphEdge(source="StartEvent", target="step1"),
            WorkflowGraphEdge(source="step1", target="resource_123", label="db"),
        ],
    )

    # Serialize
    data = graph.model_dump()
    assert len(data["nodes"]) == 4
    assert len(data["edges"]) == 2

    # Check discriminator values are present
    node_types = {n["node_type"] for n in data["nodes"]}
    assert node_types == {"step", "event", "external", "resource"}

    # Deserialize
    restored = WorkflowGraph.model_validate(data)
    assert len(restored.nodes) == 4
    assert len(restored.edges) == 2

    # Check correct types restored
    step_nodes = [n for n in restored.nodes if isinstance(n, WorkflowStepNode)]
    event_nodes = [n for n in restored.nodes if isinstance(n, WorkflowEventNode)]
    external_nodes = [n for n in restored.nodes if isinstance(n, WorkflowExternalNode)]
    resource_nodes = [n for n in restored.nodes if isinstance(n, WorkflowResourceNode)]

    assert len(step_nodes) == 1
    assert len(event_nodes) == 1
    assert len(external_nodes) == 1
    assert len(resource_nodes) == 1

    # Verify event node has its method
    assert event_nodes[0].is_subclass_of("StartEvent")

    # Verify resource node has its fields
    assert resource_nodes[0].type_name == "DatabaseClient"
    assert resource_nodes[0].getter_name == "get_db"


def test_graph_deserialization_from_raw_json() -> None:
    """Test that graph can be deserialized from raw JSON dict."""
    raw_data = {
        "nodes": [
            {"id": "step1", "label": "Step 1", "node_type": "step"},
            {
                "id": "MyEvent",
                "label": "MyEvent",
                "node_type": "event",
                "event_type": "MyEvent",
                "event_types": ["MyEvent"],
            },
            {"id": "external", "label": "External", "node_type": "external"},
            {
                "id": "resource_xyz",
                "label": "Resource",
                "node_type": "resource",
                "type_name": "SomeType",
            },
        ],
        "edges": [{"source": "MyEvent", "target": "step1"}],
    }

    graph = WorkflowGraph.model_validate(raw_data)

    assert len(graph.nodes) == 4
    assert isinstance(graph.nodes[0], WorkflowStepNode)
    assert isinstance(graph.nodes[1], WorkflowEventNode)
    assert isinstance(graph.nodes[2], WorkflowExternalNode)
    assert isinstance(graph.nodes[3], WorkflowResourceNode)
