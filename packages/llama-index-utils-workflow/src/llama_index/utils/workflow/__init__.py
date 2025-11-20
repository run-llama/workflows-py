# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union, cast

from llama_index.core.agent.workflow import (
    AgentWorkflow,
    BaseWorkflowAgent,
    CodeActAgent,
    ReActAgent,
)
from llama_index.core.tools import AsyncBaseTool, BaseTool
from pyvis.network import Network
from workflows import Workflow
from workflows.events import (
    Event,
    StartEvent,
    StopEvent,
)
from workflows.handler import WorkflowHandler
from workflows.representation_utils import (
    DrawWorkflowEdge,
    DrawWorkflowGraph,
    DrawWorkflowNode,
    _truncate_label,
)
from workflows.representation_utils import (
    extract_workflow_structure as _extract_workflow_structure,
)
from workflows.runtime.types.results import AddCollectedEvent, StepWorkerResult
from workflows.runtime.types.ticks import TickAddEvent, TickStepResult, WorkflowTick


def _get_node_color(node: DrawWorkflowNode) -> str:
    """Determine color for a node based on its type and event_type."""
    if node.node_type == "step":
        return "#ADD8E6"  # Light blue for steps
    elif node.node_type == "external":
        return "#BEDAE4"  # Light blue-gray for external
    elif node.node_type == "event" and node.event_type:
        return _determine_event_color(node.event_type)  # Uses original function
    elif node.node_type == "agent":
        # Determine color based on agent type
        if node.event_type and issubclass(node.event_type, ReActAgent):
            return "#E27AFF"
        elif node.event_type and issubclass(node.event_type, CodeActAgent):
            return "#66ccff"
        else:
            return "#90EE90"
    elif node.node_type == "tool":
        return "#ff9966"  # Orange for tools
    elif node.node_type == "workflow_base":
        return "#90EE90"  # Light green for workflow base
    elif node.node_type == "workflow_agent":
        return "#66ccff"  # Light blue for workflow agents
    elif node.node_type == "workflow_tool":
        return "#ff9966"  # Orange for workflow tools
    elif node.node_type == "workflow_handoff":
        return "#E27AFF"  # Pink for handoff nodes
    else:
        return "#90EE90"  # Default light green


def _get_node_shape(node: DrawWorkflowNode) -> str:
    """Determine shape for a node based on its type."""
    if node.node_type == "step" or node.node_type == "external":
        return "box"  # Steps and external_step use box
    elif node.node_type == "event":
        return "ellipse"  # Events use ellipse
    elif node.node_type == "agent":
        return "ellipse"  # Agents use ellipse
    elif node.node_type == "tool":
        return "ellipse"  # Tools use ellipse (matching original ellipsis behavior)
    elif node.node_type == "workflow_base":
        return "diamond"  # Workflow base uses diamond
    elif node.node_type == "workflow_agent":
        return "ellipse"  # Workflow agents use ellipse
    elif node.node_type == "workflow_tool":
        return "box"  # Workflow tools use box (matching original square behavior)
    elif node.node_type == "workflow_handoff":
        return "ellipse"  # Handoff nodes use ellipse
    else:
        return "box"  # Default shape


def _render_pyvis(
    graph: DrawWorkflowGraph, filename: str, notebook: bool = False
) -> None:
    """Render workflow graph using Pyvis."""

    net = Network(directed=True, height="750px", width="100%")

    # Add nodes
    for node in graph.nodes:
        color = _get_node_color(node)
        shape = _get_node_shape(node)
        net.add_node(
            node.id,
            label=node.label,
            title=node.title,
            color=color,
            shape=shape,
        )

    # Add edges
    for edge in graph.edges:
        net.add_edge(edge.source, edge.target)

    net.show(filename, notebook=notebook)


def _determine_event_color(event_type: type) -> str:
    """Determine color for an event type."""
    if issubclass(event_type, StartEvent):
        # Pink for start events
        event_color = "#E27AFF"
    elif issubclass(event_type, StopEvent):
        # Orange for stop events
        event_color = "#FFA07A"
    else:
        # Light green for other events
        event_color = "#90EE90"
    return event_color


def _clean_id_for_mermaid(name: str) -> str:
    """Convert a name to a valid Mermaid ID."""
    return name.replace(" ", "_").replace("-", "_").replace(".", "_")


def _get_mermaid_css_class(node: DrawWorkflowNode) -> str:
    """Determine CSS class for a node in Mermaid based on its type and event_type."""
    if node.node_type == "step":
        return "stepStyle"
    elif node.node_type == "external":
        return "externalStyle"
    elif node.node_type == "event" and node.event_type:
        if issubclass(node.event_type, StartEvent):
            return "startEventStyle"
        elif issubclass(node.event_type, StopEvent):
            return "stopEventStyle"
        else:
            return "defaultEventStyle"
    elif node.node_type == "agent":
        # Determine class based on agent type
        if node.event_type and issubclass(node.event_type, ReActAgent):
            return "reactAgentStyle"
        elif node.event_type and issubclass(node.event_type, CodeActAgent):
            return "codeActAgentStyle"
        else:
            return "defaultAgentStyle"
    elif node.node_type == "tool":
        return "toolStyle"
    elif node.node_type == "workflow_base":
        return "workflowBaseStyle"
    elif node.node_type == "workflow_agent":
        return "workflowAgentStyle"
    elif node.node_type == "workflow_tool":
        return "workflowToolStyle"
    elif node.node_type == "workflow_handoff":
        return "workflowHandoffStyle"
    else:
        return "defaultEventStyle"


def _render_mermaid(graph: DrawWorkflowGraph, filename: str) -> str:
    """Render workflow graph using Mermaid."""
    mermaid_lines = ["flowchart TD"]
    added_nodes = set()
    added_edges = set()

    # Add nodes
    for node in graph.nodes:
        # Clean ID for Mermaid
        if node.node_type == "step":
            clean_id = f"step_{_clean_id_for_mermaid(node.id)}"
        elif node.node_type == "external":
            clean_id = node.id  # external_step is already clean
        elif node.node_type in [
            "agent",
            "tool",
            "workflow_base",
            "workflow_agent",
            "workflow_tool",
            "workflow_handoff",
        ]:
            clean_id = _clean_id_for_mermaid(node.id)
        else:  # event
            clean_id = f"event_{_clean_id_for_mermaid(node.id)}"

        if clean_id not in added_nodes:
            added_nodes.add(clean_id)

            # Format node based on shape
            shape = _get_node_shape(node)
            if shape == "box":
                shape_start, shape_end = "[", "]"
            elif shape == "ellipse":
                shape_start, shape_end = "([", "])"
            elif shape == "diamond":
                shape_start, shape_end = "{", "}"
            else:
                shape_start, shape_end = "[", "]"

            css_class = _get_mermaid_css_class(node)
            mermaid_lines.append(
                f'    {clean_id}{shape_start}"{node.label}"{shape_end}:::{css_class}'
            )

    # Add edges
    for edge in graph.edges:
        source_node = next(n for n in graph.nodes if n.id == edge.source)
        target_node = next(n for n in graph.nodes if n.id == edge.target)

        if source_node.node_type == "step":
            source_id = f"step_{_clean_id_for_mermaid(edge.source)}"
        elif source_node.node_type == "external":
            source_id = edge.source
        elif source_node.node_type in [
            "agent",
            "tool",
            "workflow_base",
            "workflow_agent",
            "workflow_tool",
            "workflow_handoff",
        ]:
            source_id = _clean_id_for_mermaid(edge.source)
        else:  # event
            source_id = f"event_{_clean_id_for_mermaid(edge.source)}"

        if target_node.node_type == "step":
            target_id = f"step_{_clean_id_for_mermaid(edge.target)}"
        elif target_node.node_type == "external":
            target_id = edge.target
        elif target_node.node_type in [
            "agent",
            "tool",
            "workflow_base",
            "workflow_agent",
            "workflow_tool",
            "workflow_handoff",
        ]:
            target_id = _clean_id_for_mermaid(edge.target)
        else:  # event
            target_id = f"event_{_clean_id_for_mermaid(edge.target)}"

        edge_str = f"{source_id} --> {target_id}"
        if edge_str not in added_edges:
            added_edges.add(edge_str)
            mermaid_lines.append(f"    {edge_str}")

    # Add style definitions
    mermaid_lines.extend(
        [
            "    classDef stepStyle fill:#ADD8E6,color:#000000,line-height:1.2",
            "    classDef externalStyle fill:#BEDAE4,color:#000000,line-height:1.2",
            "    classDef startEventStyle fill:#E27AFF,color:#000000",
            "    classDef stopEventStyle fill:#FFA07A,color:#000000",
            "    classDef defaultEventStyle fill:#90EE90,color:#000000",
            "    classDef reactAgentStyle fill:#E27AFF,color:#000000",
            "    classDef codeActAgentStyle fill:#66ccff,color:#000000",
            "    classDef defaultAgentStyle fill:#90EE90,color:#000000",
            "    classDef toolStyle fill:#ff9966,color:#000000",
            "    classDef workflowBaseStyle fill:#90EE90,color:#000000",
            "    classDef workflowAgentStyle fill:#66ccff,color:#000000",
            "    classDef workflowToolStyle fill:#ff9966,color:#000000",
            "    classDef workflowHandoffStyle fill:#E27AFF,color:#000000",
        ]
    )

    diagram_string = "\n".join(mermaid_lines)

    if filename:
        with open(filename, "w") as f:
            f.write(diagram_string)

    return diagram_string


def _extract_single_agent_structure(agent: BaseWorkflowAgent) -> DrawWorkflowGraph:
    """Extract the structure of a single agent."""
    nodes = []
    edges = []

    # Add agent node
    agent_node = DrawWorkflowNode(
        id="agent",
        label=agent.name,
        node_type="agent",
        event_type=type(agent),  # Store agent type for color determination
    )
    nodes.append(agent_node)

    # Add tool nodes and edges
    tools = cast(Union[List[Union[BaseTool, AsyncBaseTool]], None], agent.tools)
    if tools is not None and len(tools) > 0:
        for i, tool in enumerate(tools):
            tool_id = f"tool_{i}"
            tool_node = DrawWorkflowNode(
                id=tool_id,
                label=f"Tool {i + 1}: {tool.metadata.get_name()}",
                node_type="tool",
            )
            nodes.append(tool_node)

            # Add edge from agent to tool
            edges.append(DrawWorkflowEdge("agent", tool_id))

    return DrawWorkflowGraph(nodes=nodes, edges=edges)


def _process_tools_and_handoffs(
    agent: BaseWorkflowAgent,
    processed_agents: List[str],
    all_agents: Dict[str, BaseWorkflowAgent],
    nodes: List[DrawWorkflowNode],
    edges: List[DrawWorkflowEdge],
    root_agent: str,
) -> Tuple[List[DrawWorkflowNode], List[DrawWorkflowEdge], List[str]]:
    if agent.name not in processed_agents:
        nodes.append(
            DrawWorkflowNode(
                id=agent.name, label=agent.name, node_type="workflow_agent"
            )
        )
        if agent.name == root_agent:
            edges.append(DrawWorkflowEdge("user", root_agent))
        for t in agent.tools or []:
            if isinstance(t, BaseTool):
                fn_name = t.metadata.get_name()
            else:
                # Fallback for non-BaseTool callables or objects without __name__
                fn_name = getattr(t, "__name__", type(t).__name__)
            node_id = f"{agent.name}_{fn_name}"
            nodes.append(
                DrawWorkflowNode(
                    id=node_id,
                    label=fn_name,
                    node_type="workflow_tool",
                )
            )
            edges.append(DrawWorkflowEdge(agent.name, node_id))
        if agent.can_handoff_to:
            for a in agent.can_handoff_to:
                edges.append(
                    DrawWorkflowEdge(
                        agent.name,
                        a,
                    )
                )
        else:
            edges.append(
                DrawWorkflowEdge(
                    agent.name,
                    "output",
                )
            )
        processed_agents.append(agent.name)

    if agent.can_handoff_to:
        for a in agent.can_handoff_to:
            if a not in processed_agents:
                _process_tools_and_handoffs(
                    all_agents[a],
                    processed_agents=processed_agents,
                    all_agents=all_agents,
                    nodes=nodes,
                    edges=edges,
                    root_agent=root_agent,
                )

    return nodes, edges, processed_agents


def _extract_agent_workflow_structure(
    agent_workflow: AgentWorkflow,
) -> DrawWorkflowGraph:
    """Extract the structure of an agent workflow."""
    nodes: List[DrawWorkflowNode] = []
    edges: List[DrawWorkflowEdge] = []

    # Add base workflow node
    user_node = DrawWorkflowNode(
        id="user",
        label="User",
        node_type="workflow_base",
    )
    output_node = DrawWorkflowNode(
        id="output", label="Output", node_type="workflow_base"
    )
    nodes.extend([user_node, output_node])

    agents = agent_workflow.agents
    processed_agents: List[str] = []
    for v in agents.values():
        nodes, edges, processed_agents = _process_tools_and_handoffs(
            agent=v,
            processed_agents=processed_agents,
            all_agents=agents,
            nodes=nodes,
            edges=edges,
            root_agent=agent_workflow.root_agent,
        )
    if all(edge.target != "output" for edge in edges):
        agent_nodes = [n for n in nodes if n.node_type == "workflow_agent"]
        edges.append(DrawWorkflowEdge(agent_nodes[-1].id, "output"))

    return DrawWorkflowGraph(nodes=nodes, edges=edges)


def draw_all_possible_flows(
    workflow: Workflow,
    filename: str = "workflow_all_flows.html",
    notebook: bool = False,
    max_label_length: int | None = None,
) -> None:
    """
    Draws all possible flows of the workflow using Pyvis.

    Args:
        workflow: The workflow to visualize
        filename: Output HTML filename
        notebook: Whether running in notebook environment
        max_label_length: Maximum label length before truncation (None = no limit)

    """
    graph = _extract_workflow_structure(workflow, max_label_length)
    _render_pyvis(graph, filename, notebook)


def draw_most_recent_execution(
    handler: WorkflowHandler,
    filename: str = "workflow_recent_execution.html",
    notebook: bool = False,
    max_label_length: int | None = None,
) -> None:
    """
    Draws the most recent execution of the workflow.

    Args:
        workflow: The workflow to visualize
        filename: Output HTML filename
        notebook: Whether running in notebook environment
        max_label_length: Maximum label length before truncation (None = no limit)

    """
    net = Network(directed=True, height="750px", width="100%")

    if handler.ctx is None or handler.ctx._broker_run is None:
        raise ValueError("No context/run info in this handler. Has it been run yet?")
    ticks: List[WorkflowTick] = handler.ctx._broker_run._tick_log

    # Build execution DAG from ticks
    nodes: Dict[str, Tuple[str, str, type | None]] = {}
    edges: List[Tuple[str, str]] = []
    event_node_by_identity: Dict[int, str] = {}
    step_seq: Dict[str, int] = {}

    # Optional external node for externally added events
    external_node_id = "external_step"
    nodes[external_node_id] = ("external_step", "external", None)

    def ensure_event_node(ev: Event) -> str:
        key = id(ev)
        if key in event_node_by_identity:
            return event_node_by_identity[key]
        label = type(ev).__name__
        node_id = f"event:{label}#{len(event_node_by_identity)}"
        # Truncate label if requested (node label only, id remains stable)
        display_label = (
            _truncate_label(label, max_label_length) if max_label_length else label
        )
        nodes[node_id] = (display_label, "event", type(ev))
        event_node_by_identity[key] = node_id
        return node_id

    # Helper to enumerate events emitted from a step result list
    def iter_emitted_events(step_tick: TickStepResult[Any]) -> List[Event]:
        emitted: List[Event] = []
        for r in step_tick.result:
            if isinstance(r, StepWorkerResult) and isinstance(r.result, Event):
                emitted.append(r.result)
            elif isinstance(r, AddCollectedEvent):
                emitted.append(r.event)
        return emitted

    for idx, t in enumerate(ticks):
        if isinstance(t, TickAddEvent):
            ev_id = ensure_event_node(t.event)
            edges.append((external_node_id, ev_id))
        elif isinstance(t, TickStepResult):
            # Create a step execution node
            step_seq[t.step_name] = step_seq.get(t.step_name, 0) + 1
            seq = step_seq[t.step_name]
            step_node_id = f"step:{t.step_name}#{seq}"
            step_label = f"{t.step_name}#{seq}"
            display_label = (
                _truncate_label(step_label, max_label_length)
                if max_label_length
                else step_label
            )
            nodes[step_node_id] = (display_label, "step", None)

            # consumed event -> step
            in_event_node_id = ensure_event_node(t.event)
            edges.append((in_event_node_id, step_node_id))

            # step -> emitted events
            for out_ev in iter_emitted_events(t):
                out_event_node_id = ensure_event_node(out_ev)
                edges.append((step_node_id, out_event_node_id))

    # Render with Pyvis
    # Add nodes first
    for node_id, (label, node_type, ev_type) in nodes.items():
        if node_type == "step":
            color = "#ADD8E6"
            shape = "box"
        elif node_type == "external":
            color = "#BEDAE4"
            shape = "box"
        else:
            # event
            color = _determine_event_color(ev_type if ev_type else Event)
            shape = "ellipse"
        net.add_node(node_id, label=label, color=color, shape=shape)

    # Then edges
    for src, dst in edges:
        net.add_edge(src, dst)

    # Suggest a hierarchical layout to preserve timeOrder-ish viewing
    try:
        net.set_options(
            '{"layout": {"hierarchical": {"enabled": true, "direction": "LR", "nodeSpacing": 150, "levelSeparation": 120}}, "physics": {"enabled": false}}'
        )
    except Exception:
        pass

    net.show(filename, notebook=notebook)


def draw_all_possible_flows_mermaid(
    workflow: Workflow,
    filename: str = "workflow_all_flows.mermaid",
    max_label_length: int | None = None,
) -> str:
    """
    Draws all possible flows of the workflow as a Mermaid diagram.

    Args:
        workflow: The workflow to visualize
        filename: Output Mermaid filename
        max_label_length: Maximum label length before truncation (None = no limit)

    Returns:
        The Mermaid diagram as a string

    """
    graph = _extract_workflow_structure(workflow, max_label_length)
    return _render_mermaid(graph, filename)


def draw_agent_with_tools(
    agent: BaseWorkflowAgent,
    filename: str = "agent_with_tools.html",
    notebook: bool = False,
) -> str:
    """
    > **NOTE**: *PyVis is needed for this function*.
    Draw an agent with its tool as a flowchart.

    Args:
        agent (BaseWorkflowAgent): agent workflow.
        filename (str): name of the HTML file to save the flowchart to. Defaults to 'agent_workflow.html'.
        notebook (bool): whether or not this is displayed within a notebook (.ipynb). Defaults to False.

    Returns:
        str: the path to the file where the flowchart was saved.

    """
    graph = _extract_single_agent_structure(agent)
    _render_pyvis(graph, filename, notebook)
    return filename


def draw_agent_workflow(
    agent_workflow: AgentWorkflow,
    filename: str = "agent_workflow.html",
    notebook: bool = False,
) -> str:
    """
    > **NOTE**: *PyVis is needed for this function*.
    Draw an agent workflow as a flowchart.

    Args:
        agent_workflow (AgentWorkflow): agent workflow.
        filename (str): name of the HTML file to save the flowchart to. Defaults to 'agent_workflow.html'.
        notebook (bool): whether or not this is displayed within a notebook (.ipynb). Defaults to False.

    Returns:
        str: the path to the file where the flowchart was saved.

    """
    graph = _extract_agent_workflow_structure(agent_workflow)
    _render_pyvis(graph, filename, notebook)
    return filename


def draw_agent_with_tools_mermaid(
    agent: BaseWorkflowAgent,
    filename: str = "agent_with_tools.mermaid",
) -> str:
    """
    Draw an agent with its tools as a Mermaid diagram.

    Args:
        agent (BaseWorkflowAgent): agent workflow.
        filename (str): name of the Mermaid file to save the diagram to. Defaults to 'agent_with_tools.mermaid'.

    Returns:
        str: the Mermaid diagram as a string.

    """
    graph = _extract_single_agent_structure(agent)
    return _render_mermaid(graph, filename)


def draw_agent_workflow_mermaid(
    agent_workflow: AgentWorkflow,
    filename: str = "agent_workflow.mermaid",
) -> str:
    """
    Draw an agent workflow as a Mermaid diagram.

    Args:
        agent_workflow (AgentWorkflow): agent workflow.
        filename (str): name of the Mermaid file to save the diagram to. Defaults to 'agent_workflow.mermaid'.

    Returns:
        str: the Mermaid diagram as a string.

    """
    graph = _extract_agent_workflow_structure(agent_workflow)
    return _render_mermaid(graph, filename)
