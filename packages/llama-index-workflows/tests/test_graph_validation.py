# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

from typing import Union

from workflows.decorators import StepConfig, StepGraphCheck
from workflows.events import (
    Event,
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
)
from workflows.representation.validate import (
    build_step_graph,
    validate_graph,
)

# -- Test event classes -------------------------------------------------------


class _GraphValidationIslandEvent(Event):
    pass


class _GraphValidationProcessedEvent(Event):
    pass


class _DeadEndCycleA(Event):
    pass


class _DeadEndCycleB(Event):
    pass


class _DeadEndLoopEvent(Event):
    pass


# -- Helper -------------------------------------------------------------------


def make_step_config(
    accepted_events: list[type],
    return_types: list[type],
    skip_graph_checks: Union[list[StepGraphCheck], None] = None,
) -> StepConfig:
    return StepConfig(
        accepted_events=accepted_events,
        event_name=accepted_events[0].__name__ if accepted_events else "unknown",
        return_types=return_types,
        context_parameter=None,
        num_workers=1,
        retry_policy=None,
        resources=[],
        skip_graph_checks=skip_graph_checks or [],
    )


# -- Tests --------------------------------------------------------------------


def test_validate_graph_empty_steps() -> None:
    errors = validate_graph(steps={}, start_event_class=StartEvent)
    assert errors == []


def test_validate_graph_valid_simple() -> None:
    steps = {
        "process": make_step_config(
            accepted_events=[StartEvent],
            return_types=[StopEvent],
        ),
    }
    errors = validate_graph(steps=steps, start_event_class=StartEvent)
    assert errors == []


def test_validate_graph_unreachable_step() -> None:
    steps = {
        "process": make_step_config(
            accepted_events=[StartEvent],
            return_types=[StopEvent],
        ),
        "island": make_step_config(
            accepted_events=[_GraphValidationIslandEvent],
            return_types=[StopEvent],
        ),
    }
    errors = validate_graph(steps=steps, start_event_class=StartEvent)
    assert len(errors) >= 1
    reachability_errors = [e for e in errors if e.check == "reachability"]
    assert len(reachability_errors) == 1
    assert "island" in reachability_errors[0].step_names
    # The island event is also terminal (produced by nobody but consumed by island,
    # however island is unreachable). The key assertion is the reachability error.


def test_validate_graph_human_response_reachable() -> None:
    """A step consuming a HumanResponseEvent subclass is reachable without StartEvent path."""

    class _MyHumanResponse(HumanResponseEvent):
        pass

    steps = {
        "start_step": make_step_config(
            accepted_events=[StartEvent],
            return_types=[InputRequiredEvent],
        ),
        "human_step": make_step_config(
            accepted_events=[_MyHumanResponse],
            return_types=[StopEvent],
        ),
    }
    errors = validate_graph(steps=steps, start_event_class=StartEvent)
    assert errors == []


def test_validate_graph_human_response_mutation_allowed() -> None:
    """A HumanResponseEvent step returning None is valid (mutation-only step)."""

    class _MyHumanResponse(HumanResponseEvent):
        pass

    steps = {
        "start_step": make_step_config(
            accepted_events=[StartEvent],
            return_types=[InputRequiredEvent],
        ),
        "human_step": make_step_config(
            accepted_events=[_MyHumanResponse],
            return_types=[type(None)],
        ),
    }
    errors = validate_graph(steps=steps, start_event_class=StartEvent)
    assert errors == []


def test_validate_graph_terminal_non_output_event() -> None:
    steps = {
        "process": make_step_config(
            accepted_events=[StartEvent],
            return_types=[_GraphValidationProcessedEvent],
        ),
    }
    errors = validate_graph(steps=steps, start_event_class=StartEvent)
    terminal_errors = [e for e in errors if e.check == "terminal_event"]
    assert len(terminal_errors) == 1
    assert "_GraphValidationProcessedEvent" in terminal_errors[0].message


def test_validate_graph_terminal_event_accumulated() -> None:
    """Multiple dangling events are reported in a single terminal_event error."""

    class _DanglingA(Event):
        pass

    class _DanglingB(Event):
        pass

    steps = {
        "process": make_step_config(
            accepted_events=[StartEvent],
            return_types=[_DanglingA, _DanglingB],
        ),
    }
    errors = validate_graph(steps=steps, start_event_class=StartEvent)
    terminal_errors = [e for e in errors if e.check == "terminal_event"]
    assert len(terminal_errors) == 1
    assert "_DanglingA" in terminal_errors[0].message
    assert "_DanglingB" in terminal_errors[0].message


def test_validate_graph_dead_end_cycle() -> None:
    """A cycle with no exit to StopEvent produces a dead_end error."""
    steps = {
        "step_a": make_step_config(
            accepted_events=[StartEvent, _DeadEndCycleB],
            return_types=[_DeadEndCycleA],
        ),
        "step_b": make_step_config(
            accepted_events=[_DeadEndCycleA],
            return_types=[_DeadEndCycleB],
        ),
    }
    errors = validate_graph(steps=steps, start_event_class=StartEvent)
    dead_end_errors = [e for e in errors if e.check == "dead_end"]
    assert len(dead_end_errors) == 1
    assert "step_a" in dead_end_errors[0].step_names
    assert "step_b" in dead_end_errors[0].step_names


def test_validate_graph_dead_end_with_exit_branch_passes() -> None:
    """A cycle where one branch reaches StopEvent passes validation."""
    steps = {
        "step_a": make_step_config(
            accepted_events=[StartEvent, _DeadEndCycleB],
            return_types=[_DeadEndCycleA, StopEvent],
        ),
        "step_b": make_step_config(
            accepted_events=[_DeadEndCycleA],
            return_types=[_DeadEndCycleB],
        ),
    }
    errors = validate_graph(steps=steps, start_event_class=StartEvent)
    assert errors == []


def test_validate_graph_skip_reachability_per_step() -> None:
    steps = {
        "process": make_step_config(
            accepted_events=[StartEvent],
            return_types=[StopEvent],
        ),
        "island": make_step_config(
            accepted_events=[_GraphValidationIslandEvent],
            return_types=[StopEvent],
            skip_graph_checks=["reachability"],
        ),
    }
    errors = validate_graph(steps=steps, start_event_class=StartEvent)
    reachability_errors = [e for e in errors if e.check == "reachability"]
    assert len(reachability_errors) == 0


def test_validate_graph_skip_reachability_workflow_level() -> None:
    steps = {
        "process": make_step_config(
            accepted_events=[StartEvent],
            return_types=[StopEvent],
        ),
        "island": make_step_config(
            accepted_events=[_GraphValidationIslandEvent],
            return_types=[StopEvent],
        ),
    }
    errors = validate_graph(
        steps=steps, start_event_class=StartEvent, skip_checks={"reachability"}
    )
    reachability_errors = [e for e in errors if e.check == "reachability"]
    assert len(reachability_errors) == 0


def test_validate_graph_skip_terminal_event_workflow_level() -> None:
    steps = {
        "process": make_step_config(
            accepted_events=[StartEvent],
            return_types=[_GraphValidationProcessedEvent],
        ),
    }
    errors = validate_graph(
        steps=steps, start_event_class=StartEvent, skip_checks={"terminal_event"}
    )
    terminal_errors = [e for e in errors if e.check == "terminal_event"]
    assert len(terminal_errors) == 0


def test_validate_graph_skip_dead_end_per_step() -> None:
    steps = {
        "step_a": make_step_config(
            accepted_events=[StartEvent, _DeadEndCycleB],
            return_types=[_DeadEndCycleA],
            skip_graph_checks=["dead_end"],
        ),
        "step_b": make_step_config(
            accepted_events=[_DeadEndCycleA],
            return_types=[_DeadEndCycleB],
            skip_graph_checks=["dead_end"],
        ),
    }
    errors = validate_graph(steps=steps, start_event_class=StartEvent)
    dead_end_errors = [e for e in errors if e.check == "dead_end"]
    assert len(dead_end_errors) == 0


def test_validate_graph_skip_dead_end_workflow_level() -> None:
    steps = {
        "step_a": make_step_config(
            accepted_events=[StartEvent, _DeadEndCycleB],
            return_types=[_DeadEndCycleA],
        ),
        "step_b": make_step_config(
            accepted_events=[_DeadEndCycleA],
            return_types=[_DeadEndCycleB],
        ),
    }
    errors = validate_graph(
        steps=steps, start_event_class=StartEvent, skip_checks={"dead_end"}
    )
    dead_end_errors = [e for e in errors if e.check == "dead_end"]
    assert len(dead_end_errors) == 0


def test_validate_graph_multiple_errors_accumulated() -> None:
    """A graph that fails both reachability and dead-end returns 2+ errors."""
    steps = {
        # Unreachable island step
        "island": make_step_config(
            accepted_events=[_GraphValidationIslandEvent],
            return_types=[StopEvent],
        ),
        # Dead-end cycle (reachable from StartEvent but no path to StopEvent)
        "cycle_a": make_step_config(
            accepted_events=[StartEvent, _DeadEndCycleB],
            return_types=[_DeadEndCycleA],
        ),
        "cycle_b": make_step_config(
            accepted_events=[_DeadEndCycleA],
            return_types=[_DeadEndCycleB],
        ),
    }
    errors = validate_graph(steps=steps, start_event_class=StartEvent)
    checks_found = {e.check for e in errors}
    assert "reachability" in checks_found
    assert "dead_end" in checks_found
    assert len(errors) >= 2


# -- build_step_graph tests ---------------------------------------------------


def test_build_step_graph_empty() -> None:
    graph = build_step_graph(steps={}, start_event_class=StartEvent)
    assert graph.step_names == set()
    assert graph.outgoing == {}
    assert graph.event_types == set()
    assert StartEvent in graph.forward_reachable  # seed is always present


def test_build_step_graph_adjacency_list() -> None:
    """Adjacency list has edges: StartEvent -> step_a, step_a -> ProcessedEvent,
    ProcessedEvent -> step_b, step_b -> StopEvent."""
    steps = {
        "step_a": make_step_config(
            accepted_events=[StartEvent],
            return_types=[_GraphValidationProcessedEvent],
        ),
        "step_b": make_step_config(
            accepted_events=[_GraphValidationProcessedEvent],
            return_types=[StopEvent],
        ),
    }
    graph = build_step_graph(steps, start_event_class=StartEvent)

    assert graph.step_names == {"step_a", "step_b"}
    assert "step_a" in graph.outgoing[StartEvent]
    assert _GraphValidationProcessedEvent in graph.outgoing["step_a"]
    assert "step_b" in graph.outgoing[_GraphValidationProcessedEvent]
    assert StopEvent in graph.outgoing["step_b"]


def test_build_step_graph_event_types() -> None:
    steps = {
        "step_a": make_step_config(
            accepted_events=[StartEvent],
            return_types=[_DeadEndCycleA, StopEvent],
        ),
    }
    graph = build_step_graph(steps, start_event_class=StartEvent)

    assert graph.event_types == {StartEvent, _DeadEndCycleA, StopEvent}


def test_build_step_graph_none_return_type_excluded() -> None:
    """Steps returning None should not add NoneType to the adjacency list."""
    steps = {
        "step_a": make_step_config(
            accepted_events=[StartEvent],
            return_types=[type(None)],
        ),
    }
    graph = build_step_graph(steps, start_event_class=StartEvent)

    assert type(None) not in graph.event_types
    assert "step_a" not in graph.outgoing  # no outgoing edges from step_a


def test_build_step_graph_forward_reachable() -> None:
    """Forward reachable includes start seed, all steps and events on the path."""
    steps = {
        "step_a": make_step_config(
            accepted_events=[StartEvent],
            return_types=[_GraphValidationProcessedEvent],
        ),
        "step_b": make_step_config(
            accepted_events=[_GraphValidationProcessedEvent],
            return_types=[StopEvent],
        ),
        "island": make_step_config(
            accepted_events=[_GraphValidationIslandEvent],
            return_types=[StopEvent],
        ),
    }
    graph = build_step_graph(steps, start_event_class=StartEvent)

    assert "step_a" in graph.forward_reachable
    assert "step_b" in graph.forward_reachable
    assert _GraphValidationProcessedEvent in graph.forward_reachable
    assert "island" not in graph.forward_reachable
    assert _GraphValidationIslandEvent not in graph.forward_reachable


def test_build_step_graph_forward_reachable_human_response_seed() -> None:
    """HumanResponseEvent subclasses act as additional forward-reachability seeds."""

    class _TestHumanResponse(HumanResponseEvent):
        pass

    steps = {
        "start_step": make_step_config(
            accepted_events=[StartEvent],
            return_types=[InputRequiredEvent],
        ),
        "human_step": make_step_config(
            accepted_events=[_TestHumanResponse],
            return_types=[StopEvent],
        ),
    }
    graph = build_step_graph(steps, start_event_class=StartEvent)

    assert "human_step" in graph.forward_reachable
    assert _TestHumanResponse in graph.forward_reachable


def test_build_step_graph_reverse_reachable() -> None:
    """Reverse reachable includes steps/events on a path back from output events."""
    steps = {
        "step_a": make_step_config(
            accepted_events=[StartEvent],
            return_types=[_GraphValidationProcessedEvent],
        ),
        "step_b": make_step_config(
            accepted_events=[_GraphValidationProcessedEvent],
            return_types=[StopEvent],
        ),
        "dead_end": make_step_config(
            accepted_events=[StartEvent],
            return_types=[_DeadEndCycleA],
        ),
        "loop": make_step_config(
            accepted_events=[_DeadEndCycleA],
            return_types=[_DeadEndCycleA],
        ),
    }
    graph = build_step_graph(steps, start_event_class=StartEvent)

    # step_a and step_b can reach StopEvent
    assert "step_a" in graph.reverse_reachable
    assert "step_b" in graph.reverse_reachable
    # dead_end and loop cannot reach any output event
    assert "dead_end" not in graph.reverse_reachable
    assert "loop" not in graph.reverse_reachable
