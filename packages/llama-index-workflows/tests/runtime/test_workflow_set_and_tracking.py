# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for WorkflowSet, runtime tracking, and workflow mutation."""

from __future__ import annotations

import gc
from typing import Any

import pytest
from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent
from workflows.plugins import BasicRuntime
from workflows.runtime.types.plugin import WorkflowSet


class SimpleWorkflow(Workflow):
    @step
    async def start(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="done")


class UnhashableWorkflow(Workflow):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.data = [1, 2, 3]  # Makes it unhashable

    @step
    async def start(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="done")


@pytest.fixture
def basic_runtime() -> BasicRuntime:
    return BasicRuntime()


@pytest.fixture
def workflow_set() -> WorkflowSet:
    return WorkflowSet()


# ---------------------------------------------------------------------------
# WorkflowSet tests
# ---------------------------------------------------------------------------


def test_workflow_set_add_and_contains(
    workflow_set: WorkflowSet, basic_runtime: BasicRuntime
) -> None:
    wf = SimpleWorkflow(runtime=basic_runtime)
    workflow_set.add(wf)
    assert wf in workflow_set


def test_workflow_set_add_unhashable_workflow(
    workflow_set: WorkflowSet, basic_runtime: BasicRuntime
) -> None:
    wf = UnhashableWorkflow(runtime=basic_runtime)
    workflow_set.add(wf)
    assert wf in workflow_set
    items = list(workflow_set)
    assert wf in items


def test_workflow_set_discard(
    workflow_set: WorkflowSet, basic_runtime: BasicRuntime
) -> None:
    wf = SimpleWorkflow(runtime=basic_runtime)
    workflow_set.add(wf)
    assert wf in workflow_set
    workflow_set.discard(wf)
    assert wf not in workflow_set


def test_workflow_set_len_and_bool(
    workflow_set: WorkflowSet, basic_runtime: BasicRuntime
) -> None:
    assert not workflow_set
    assert len(workflow_set) == 0

    wf = SimpleWorkflow(runtime=basic_runtime)
    workflow_set.add(wf)
    assert workflow_set
    assert len(workflow_set) == 1


def test_workflow_set_iter(
    workflow_set: WorkflowSet, basic_runtime: BasicRuntime
) -> None:
    wf1 = SimpleWorkflow(runtime=basic_runtime)
    wf2 = SimpleWorkflow(runtime=basic_runtime)
    wf3 = SimpleWorkflow(runtime=basic_runtime)
    workflow_set.add(wf1)
    workflow_set.add(wf2)
    workflow_set.add(wf3)
    items = set(id(w) for w in workflow_set)
    assert items == {id(wf1), id(wf2), id(wf3)}


def test_workflow_set_gc_cleanup(
    workflow_set: WorkflowSet, basic_runtime: BasicRuntime
) -> None:
    wf = SimpleWorkflow(runtime=basic_runtime)
    workflow_set.add(wf)
    assert len(workflow_set) == 1
    del wf
    gc.collect()
    assert len(workflow_set) == 0


def test_workflow_set_add_idempotent(
    workflow_set: WorkflowSet, basic_runtime: BasicRuntime
) -> None:
    wf = SimpleWorkflow(runtime=basic_runtime)
    workflow_set.add(wf)
    workflow_set.add(wf)
    assert len(workflow_set) == 1


# ---------------------------------------------------------------------------
# Runtime tracking tests
# ---------------------------------------------------------------------------


def test_track_workflow_adds_to_set(basic_runtime: BasicRuntime) -> None:
    wf = SimpleWorkflow(runtime=basic_runtime)
    assert wf in basic_runtime._pending


def test_untrack_workflow_removes_from_set(basic_runtime: BasicRuntime) -> None:
    wf = SimpleWorkflow(runtime=basic_runtime)
    assert wf in basic_runtime._pending
    basic_runtime.untrack_workflow(wf)
    assert wf not in basic_runtime._pending


def test_launch_locks_tracked_workflows(basic_runtime: BasicRuntime) -> None:
    wf1 = SimpleWorkflow(runtime=basic_runtime)
    wf2 = SimpleWorkflow(runtime=basic_runtime)
    basic_runtime.launch()
    assert wf1._runtime_locked is True
    assert wf2._runtime_locked is True


def test_relaunch_locks_new_workflows(basic_runtime: BasicRuntime) -> None:
    wf1 = SimpleWorkflow(runtime=basic_runtime)
    basic_runtime.launch()
    assert wf1._runtime_locked is True

    wf2 = SimpleWorkflow(runtime=basic_runtime)
    assert wf2._runtime_locked is False
    basic_runtime.launch()
    assert wf1._runtime_locked is True
    assert wf2._runtime_locked is True


def test_weak_reference_cleanup(basic_runtime: BasicRuntime) -> None:
    wf = SimpleWorkflow(runtime=basic_runtime)
    assert len(basic_runtime._pending) == 1
    del wf
    gc.collect()
    assert len(basic_runtime._pending) == 0


def test_basic_runtime_launch_sets_launched_flag(basic_runtime: BasicRuntime) -> None:
    assert basic_runtime._launched is False
    basic_runtime.launch()
    assert basic_runtime._launched is True


# ---------------------------------------------------------------------------
# Workflow mutation tests
# ---------------------------------------------------------------------------


def test_workflow_name_setter(basic_runtime: BasicRuntime) -> None:
    wf = SimpleWorkflow(runtime=basic_runtime)
    wf._switch_workflow_name("custom-name")
    assert wf.workflow_name == "custom-name"


def test_workflow_name_setter_raises_after_launch() -> None:
    rt = BasicRuntime()
    wf = SimpleWorkflow(runtime=rt)
    wf._switch_workflow_name("before-launch")
    assert wf.workflow_name == "before-launch"

    rt.launch()
    with pytest.raises(RuntimeError, match="Cannot change workflow_name"):
        wf._switch_workflow_name("after-launch")


def test_runtime_setter_swaps_tracking() -> None:
    rt1 = BasicRuntime()
    rt2 = BasicRuntime()
    wf = SimpleWorkflow(runtime=rt1)
    assert wf in rt1._pending
    assert wf not in rt2._pending

    wf._switch_runtime(rt2)
    assert wf not in rt1._pending
    assert wf in rt2._pending


def test_runtime_setter_post_launch_raises() -> None:
    rt1 = BasicRuntime()
    rt2 = BasicRuntime()
    wf = SimpleWorkflow(runtime=rt1)
    rt1.launch()

    with pytest.raises(RuntimeError, match="Cannot reassign runtime"):
        wf._switch_runtime(rt2)


def test_runtime_setter_same_runtime_after_launch_is_noop() -> None:
    rt = BasicRuntime()
    wf = SimpleWorkflow(runtime=rt)
    rt.launch()
    # Assigning the same runtime should not raise
    wf._switch_runtime(rt)
    assert wf.runtime is rt


def test_runtime_setter_before_launch_then_launch_locks() -> None:
    rt1 = BasicRuntime()
    rt2 = BasicRuntime()
    wf = SimpleWorkflow(runtime=rt1)
    wf._switch_runtime(rt2)
    assert wf._runtime_locked is False
    rt2.launch()
    assert wf._runtime_locked is True
