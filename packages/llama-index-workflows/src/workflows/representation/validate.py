# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

from dataclasses import dataclass, field

from workflows.decorators import StepConfig, WorkflowGraphCheck
from workflows.events import HumanResponseEvent, InputRequiredEvent, StopEvent

# Graph nodes: step names (str) for steps, event classes (type) for events.
GraphNode = str | type


@dataclass
class StepGraph:
    """Lightweight adjacency-list representation of a workflow's step/event graph.

    Nodes are step names (``str``) for steps and event classes (``type``) for
    events.  An edge from an event node to a step node means the step accepts
    that event; an edge from a step node to an event node means the step returns
    that event type.
    """

    outgoing: dict[GraphNode, list[GraphNode]] = field(default_factory=dict)
    """Adjacency list: node -> list of successor nodes."""

    event_types: set[type] = field(default_factory=set)
    """All event classes seen in the graph."""

    step_names: set[str] = field(default_factory=set)
    """Names of all steps in the graph."""

    forward_reachable: set[GraphNode] = field(default_factory=set)
    """Nodes reachable from input seeds (StartEvent, HumanResponseEvent subclasses)."""

    reverse_reachable: set[GraphNode] = field(default_factory=set)
    """Nodes that can reach an output event (StopEvent, InputRequiredEvent) via reverse traversal."""


def build_step_graph(
    steps: dict[str, StepConfig],
    start_event_class: type,
) -> StepGraph:
    """Build a StepGraph from step configs and a start event class.

    Constructs the adjacency list, then computes forward reachability from input
    events (StartEvent + HumanResponseEvent subclasses) and reverse reachability
    from output events (StopEvent + InputRequiredEvent).
    """
    outgoing: dict[GraphNode, list[GraphNode]] = {}
    event_types: set[type] = set()
    step_names: set[str] = set()

    for name, cfg in steps.items():
        step_names.add(name)
        for ev in cfg.accepted_events:
            event_types.add(ev)
            outgoing.setdefault(ev, []).append(name)
        for rt in cfg.return_types:
            if rt is type(None):
                continue
            event_types.add(rt)
            outgoing.setdefault(name, []).append(rt)

    # Forward DFS from StartEvent + HumanResponseEvent subclasses
    seeds: list[GraphNode] = [start_event_class]
    for ev_type in event_types:
        if issubclass(ev_type, HumanResponseEvent) and ev_type not in seeds:
            seeds.append(ev_type)

    forward_reachable = _dfs(seeds, outgoing)

    # Reverse DFS from output events
    incoming: dict[GraphNode, list[GraphNode]] = {}
    for source, targets in outgoing.items():
        for target in targets:
            incoming.setdefault(target, []).append(source)

    output_seeds: list[GraphNode] = [
        ev_type
        for ev_type in event_types
        if issubclass(ev_type, (StopEvent, InputRequiredEvent))
    ]
    reverse_reachable = _dfs(output_seeds, incoming)

    return StepGraph(
        outgoing=outgoing,
        event_types=event_types,
        step_names=step_names,
        forward_reachable=forward_reachable,
        reverse_reachable=reverse_reachable,
    )


def _dfs(
    seeds: list[GraphNode], adjacency: dict[GraphNode, list[GraphNode]]
) -> set[GraphNode]:
    """Depth-first search returning all reachable nodes from seeds."""
    visited: set[GraphNode] = set()
    stack = list(seeds)
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for target in adjacency.get(node, []):
            if target not in visited:
                stack.append(target)
    return visited


@dataclass
class GraphValidationError:
    """A single graph validation error."""

    check: WorkflowGraphCheck
    message: str
    hint: str
    step_names: list[str] = field(default_factory=list)


def validate_graph(
    steps: dict[str, StepConfig],
    start_event_class: type,
    skip_checks: set[WorkflowGraphCheck] | None = None,
) -> list[GraphValidationError]:
    """Validate the graph structure of a workflow, accumulating all errors.

    Builds a ``StepGraph`` from step configs and runs three checks:
    1. Reachability: all steps are reachable from input events
    2. Terminal events: events with no consumer must be output events
    3. Dead ends: every step producing events must reach an output event

    Args:
        steps: Mapping of step name to StepConfig.
        start_event_class: The StartEvent subclass for this workflow.
        skip_checks: Workflow-level checks to skip entirely.

    Returns:
        List of GraphValidationError (empty if the graph is valid).
    """
    skip_checks = skip_checks or set()
    errors: list[GraphValidationError] = []

    graph = build_step_graph(steps, start_event_class)

    # Check 1: Reachability
    if "reachability" not in skip_checks:
        step_skip = {
            name
            for name, cfg in steps.items()
            if "reachability" in cfg.skip_graph_checks
        }
        unreachable_steps = sorted(
            name
            for name in graph.step_names - step_skip
            if name not in graph.forward_reachable
        )
        if unreachable_steps:
            names = ", ".join(unreachable_steps)
            errors.append(
                GraphValidationError(
                    check="reachability",
                    message=f"Unreachable steps: {names}",
                    hint="Steps must be reachable from StartEvent or HumanResponseEvent.",
                    step_names=unreachable_steps,
                )
            )

    # Check 2: Terminal events — events with no step consumer must be output events
    if "terminal_event" not in skip_checks:
        dangling: list[type] = []
        for ev_type in graph.event_types:
            targets = graph.outgoing.get(ev_type, [])
            if any(t in graph.step_names for t in targets):
                continue
            if issubclass(ev_type, (StopEvent, InputRequiredEvent)):
                continue
            dangling.append(ev_type)
        if dangling:
            names = ", ".join(sorted(t.__name__ for t in dangling))
            errors.append(
                GraphValidationError(
                    check="terminal_event",
                    message=f"Events produced but never consumed: {names}",
                    hint="Only StopEvent and InputRequiredEvent may be terminal.",
                    step_names=[],
                )
            )

    # Check 3: Dead-end detection
    if "dead_end" not in skip_checks:
        steps_producing_events = {
            s
            for s in graph.step_names
            if any(isinstance(t, type) for t in graph.outgoing.get(s, []))
        }

        step_skip = {
            name for name, cfg in steps.items() if "dead_end" in cfg.skip_graph_checks
        }
        dead_end_steps = sorted(
            name
            for name in steps_producing_events - step_skip
            if name not in graph.reverse_reachable
        )
        if dead_end_steps:
            names = ", ".join(dead_end_steps)
            errors.append(
                GraphValidationError(
                    check="dead_end",
                    message=f"Dead-end steps: {names}",
                    hint="Steps must have a path to StopEvent or InputRequiredEvent.",
                    step_names=dead_end_steps,
                )
            )

    return errors
