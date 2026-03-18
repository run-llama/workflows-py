# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

from workflows.decorators import WorkflowGraphCheck, step
from workflows.events import (
    Event,
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
)
from workflows.representation.validate import (
    GraphValidationError,
    build_step_graph,
    validate_graph,
)
from workflows.workflow import Workflow

# -- Helpers ------------------------------------------------------------------


def _validate(
    wf: Workflow,
    skip_checks: set[WorkflowGraphCheck] | None = None,
) -> list[GraphValidationError]:
    step_configs = {name: func._step_config for name, func in wf._get_steps().items()}
    return validate_graph(
        steps=step_configs,
        start_event_class=wf._start_event_class,
        skip_checks=skip_checks,
    )


def _errors_by_check(
    errors: list[GraphValidationError], check: WorkflowGraphCheck
) -> list[GraphValidationError]:
    return [e for e in errors if e.check == check]


# -- Event classes ------------------------------------------------------------


class IslandEvent(Event):
    pass


class ProcessedEvent(Event):
    pass


class CycleA(Event):
    pass


class CycleB(Event):
    pass


class LoopEvent(Event):
    pass


# -- Tests: validate_graph ----------------------------------------------------


def test_validate_simple_valid() -> None:
    class Simple(Workflow):
        @step
        async def process(self, ev: StartEvent) -> StopEvent:
            return StopEvent(result="done")

    assert _validate(Simple()) == []


def test_validate_unreachable_step() -> None:
    class Unreachable(Workflow):
        @step
        async def process(self, ev: StartEvent) -> StopEvent:
            return StopEvent(result="done")

        @step
        async def island(self, ev: IslandEvent) -> StopEvent:
            return StopEvent(result="done")

    errors = _validate(Unreachable())
    reachability = _errors_by_check(errors, "reachability")
    assert len(reachability) == 1
    assert "island" in reachability[0].step_names


def test_validate_human_response_reachable() -> None:
    class MyHumanResponse(HumanResponseEvent):
        pass

    class HumanLoop(Workflow):
        @step
        async def start_step(self, ev: StartEvent) -> InputRequiredEvent:
            return InputRequiredEvent()

        @step
        async def human_step(self, ev: MyHumanResponse) -> StopEvent:
            return StopEvent(result="done")

    assert _validate(HumanLoop()) == []


def test_validate_human_response_mutation_allowed() -> None:
    """A HumanResponseEvent step returning None is valid (mutation-only)."""

    class MyHumanResponse(HumanResponseEvent):
        pass

    class MutationOnly(Workflow):
        @step
        async def start_step(self, ev: StartEvent) -> InputRequiredEvent:
            return InputRequiredEvent()

        @step
        async def human_step(self, ev: MyHumanResponse) -> StopEvent:
            return StopEvent(result="done")

        @step
        async def mutation(self, ev: MyHumanResponse) -> None:
            pass

    assert _validate(MutationOnly()) == []


def test_validate_terminal_non_output_event() -> None:
    class Dangling(Workflow):
        @step
        async def process(self, ev: StartEvent) -> ProcessedEvent | StopEvent:
            return ProcessedEvent()

    errors = _validate(Dangling())
    terminal = _errors_by_check(errors, "terminal_event")
    assert len(terminal) == 1
    assert "ProcessedEvent" in terminal[0].message


def test_validate_terminal_event_accumulated() -> None:
    """Multiple dangling events in a single terminal_event error."""

    class DanglingA(Event):
        pass

    class DanglingB(Event):
        pass

    class MultiDangling(Workflow):
        @step
        async def process(self, ev: StartEvent) -> DanglingA | DanglingB | StopEvent:
            return DanglingA()

    errors = _validate(MultiDangling())
    terminal = _errors_by_check(errors, "terminal_event")
    assert len(terminal) == 1
    assert (
        terminal[0].message
        == "Events produced but never consumed: DanglingA, DanglingB"
    )


def test_validate_dead_end_cycle() -> None:
    """A cycle with no exit to StopEvent produces a dead_end error."""

    class DeadEndCycle(Workflow):
        @step
        async def entry(self, ev: StartEvent) -> CycleA | StopEvent:
            return CycleA()

        @step
        async def step_b(self, ev: CycleA) -> CycleB:
            return CycleB()

        @step
        async def step_c(self, ev: CycleB) -> CycleA:
            return CycleA()

    errors = _validate(DeadEndCycle())
    dead_end = _errors_by_check(errors, "dead_end")
    assert len(dead_end) == 1
    # entry has a StopEvent branch so it's not a dead end, but step_b and step_c are
    assert "step_b" in dead_end[0].step_names
    assert "step_c" in dead_end[0].step_names


def test_validate_dead_end_with_exit_branch_passes() -> None:
    """A cycle where one branch reaches StopEvent passes."""

    class CycleWithExit(Workflow):
        @step
        async def step_a(self, ev: StartEvent) -> CycleA | StopEvent:
            return CycleA()

        @step
        async def step_b(self, ev: CycleA) -> CycleB | StopEvent:
            return CycleB()

        @step
        async def step_c(self, ev: CycleB) -> CycleA:
            return CycleA()

    assert _validate(CycleWithExit()) == []


def test_validate_skip_reachability_per_step() -> None:
    class SkipReach(Workflow):
        @step
        async def process(self, ev: StartEvent) -> StopEvent:
            return StopEvent(result="done")

        @step(skip_graph_checks=["reachability"])
        async def island(self, ev: IslandEvent) -> StopEvent:
            return StopEvent(result="done")

    errors = _validate(SkipReach())
    assert _errors_by_check(errors, "reachability") == []


def test_validate_skip_reachability_workflow_level() -> None:
    class WithIsland(Workflow):
        @step
        async def process(self, ev: StartEvent) -> StopEvent:
            return StopEvent(result="done")

        @step
        async def island(self, ev: IslandEvent) -> StopEvent:
            return StopEvent(result="done")

    errors = _validate(WithIsland(), skip_checks={"reachability"})
    assert _errors_by_check(errors, "reachability") == []


def test_validate_skip_terminal_event_workflow_level() -> None:
    class DanglingWf(Workflow):
        @step
        async def process(self, ev: StartEvent) -> ProcessedEvent | StopEvent:
            return ProcessedEvent()

    errors = _validate(DanglingWf(), skip_checks={"terminal_event"})
    assert _errors_by_check(errors, "terminal_event") == []


def test_validate_skip_dead_end_per_step() -> None:
    class SkipDeadEnd(Workflow):
        @step
        async def entry(self, ev: StartEvent) -> CycleA | StopEvent:
            return CycleA()

        @step(skip_graph_checks=["dead_end"])
        async def step_b(self, ev: CycleA) -> CycleB:
            return CycleB()

        @step(skip_graph_checks=["dead_end"])
        async def step_c(self, ev: CycleB) -> CycleA:
            return CycleA()

    errors = _validate(SkipDeadEnd())
    assert _errors_by_check(errors, "dead_end") == []


def test_validate_skip_dead_end_workflow_level() -> None:
    class DeadEndWf(Workflow):
        @step
        async def entry(self, ev: StartEvent) -> CycleA | StopEvent:
            return CycleA()

        @step
        async def step_b(self, ev: CycleA) -> CycleB:
            return CycleB()

        @step
        async def step_c(self, ev: CycleB) -> CycleA:
            return CycleA()

    errors = _validate(DeadEndWf(), skip_checks={"dead_end"})
    assert _errors_by_check(errors, "dead_end") == []


def test_validate_multiple_errors_accumulated() -> None:
    """A graph that fails both reachability and dead-end returns 2+ errors."""

    class MultiError(Workflow):
        @step
        async def cycle_a(self, ev: StartEvent) -> CycleA | StopEvent:
            return CycleA()

        @step
        async def cycle_b(self, ev: CycleA) -> CycleB:
            return CycleB()

        @step
        async def cycle_c(self, ev: CycleB) -> CycleA:
            return CycleA()

        @step
        async def island(self, ev: IslandEvent) -> StopEvent:
            return StopEvent(result="done")

    errors = _validate(MultiError())
    detail = "\n".join(f"  - [{e.check}] {e.message}\n    {e.hint}" for e in errors)
    msg = f"Graph validation failed:\n{detail}"
    assert (
        msg
        == """\
Graph validation failed:
  - [reachability] Unreachable steps: island
    Steps must be reachable from StartEvent or HumanResponseEvent.
  - [dead_end] Dead-end steps: cycle_b, cycle_c
    Steps must have a path to StopEvent or InputRequiredEvent."""
    )


# -- Tests: build_step_graph -------------------------------------------------


def test_build_step_graph_empty() -> None:
    graph = build_step_graph(steps={}, start_event_class=StartEvent)
    assert graph.step_names == set()
    assert graph.outgoing == {}
    assert graph.event_types == set()
    assert StartEvent in graph.forward_reachable


def test_build_step_graph_adjacency_list() -> None:
    class Chain(Workflow):
        @step
        async def step_a(self, ev: StartEvent) -> ProcessedEvent:
            return ProcessedEvent()

        @step
        async def step_b(self, ev: ProcessedEvent) -> StopEvent:
            return StopEvent(result="done")

    wf = Chain()
    step_configs = {name: func._step_config for name, func in wf._get_steps().items()}
    graph = build_step_graph(step_configs, start_event_class=StartEvent)

    assert graph.step_names == {"step_a", "step_b"}
    assert "step_a" in graph.outgoing[StartEvent]
    assert ProcessedEvent in graph.outgoing["step_a"]
    assert "step_b" in graph.outgoing[ProcessedEvent]
    assert StopEvent in graph.outgoing["step_b"]


def test_build_step_graph_event_types() -> None:
    class WithEvents(Workflow):
        @step
        async def step_a(self, ev: StartEvent) -> CycleA | StopEvent:
            return CycleA()

    wf = WithEvents()
    step_configs = {name: func._step_config for name, func in wf._get_steps().items()}
    graph = build_step_graph(step_configs, start_event_class=StartEvent)

    assert graph.event_types == {StartEvent, CycleA, StopEvent}


def test_build_step_graph_none_return_type_excluded() -> None:
    """Steps returning None should not add NoneType to the adjacency list."""

    class NoneReturn(Workflow):
        @step
        async def step_a(self, ev: StartEvent) -> StopEvent:
            return StopEvent(result="done")

        @step
        async def mutation(self, ev: StartEvent) -> None:
            pass

    wf = NoneReturn()
    step_configs = {name: func._step_config for name, func in wf._get_steps().items()}
    graph = build_step_graph(step_configs, start_event_class=StartEvent)

    assert type(None) not in graph.event_types
    assert "mutation" not in graph.outgoing


def test_build_step_graph_forward_reachable() -> None:
    class WithIsland(Workflow):
        @step
        async def step_a(self, ev: StartEvent) -> ProcessedEvent:
            return ProcessedEvent()

        @step
        async def step_b(self, ev: ProcessedEvent) -> StopEvent:
            return StopEvent(result="done")

        @step
        async def island(self, ev: IslandEvent) -> StopEvent:
            return StopEvent(result="done")

    wf = WithIsland()
    step_configs = {name: func._step_config for name, func in wf._get_steps().items()}
    graph = build_step_graph(step_configs, start_event_class=StartEvent)

    assert "step_a" in graph.forward_reachable
    assert "step_b" in graph.forward_reachable
    assert ProcessedEvent in graph.forward_reachable
    assert "island" not in graph.forward_reachable
    assert IslandEvent not in graph.forward_reachable


def test_build_step_graph_forward_reachable_human_response_seed() -> None:
    """HumanResponseEvent subclasses act as additional forward-reachability seeds."""

    class TestHumanResponse(HumanResponseEvent):
        pass

    class HumanLoop(Workflow):
        @step
        async def start_step(self, ev: StartEvent) -> InputRequiredEvent:
            return InputRequiredEvent()

        @step
        async def human_step(self, ev: TestHumanResponse) -> StopEvent:
            return StopEvent(result="done")

    wf = HumanLoop()
    step_configs = {name: func._step_config for name, func in wf._get_steps().items()}
    graph = build_step_graph(step_configs, start_event_class=StartEvent)

    assert "human_step" in graph.forward_reachable
    assert TestHumanResponse in graph.forward_reachable


def test_build_step_graph_reverse_reachable() -> None:
    class WithDeadEnd(Workflow):
        @step
        async def step_a(self, ev: StartEvent) -> ProcessedEvent:
            return ProcessedEvent()

        @step
        async def step_b(self, ev: ProcessedEvent) -> StopEvent:
            return StopEvent(result="done")

        @step
        async def dead_end(self, ev: StartEvent) -> LoopEvent:
            return LoopEvent()

        @step
        async def loop(self, ev: LoopEvent) -> LoopEvent:
            return LoopEvent()

    wf = WithDeadEnd()
    step_configs = {name: func._step_config for name, func in wf._get_steps().items()}
    graph = build_step_graph(step_configs, start_event_class=StartEvent)

    assert "step_a" in graph.reverse_reachable
    assert "step_b" in graph.reverse_reachable
    assert "dead_end" not in graph.reverse_reachable
    assert "loop" not in graph.reverse_reachable
