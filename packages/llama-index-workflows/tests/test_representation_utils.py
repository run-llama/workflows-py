import pytest
from workflows.events import StartEvent, StopEvent
from workflows.representation_utils import (
    DrawWorkflowEdge,
    DrawWorkflowGraph,
    DrawWorkflowNode,
    extract_workflow_structure,
)

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
