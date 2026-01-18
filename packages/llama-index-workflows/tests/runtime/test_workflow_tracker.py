# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for WorkflowTracker with strong references."""

from __future__ import annotations

import gc
import weakref

import pytest
from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent
from workflows.runtime.types.plugin import RegisteredWorkflow
from workflows.runtime.workflow_tracker import WorkflowTracker


class SimpleWorkflow(Workflow):
    @step
    async def start(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="done")


def test_add_workflow_to_tracker() -> None:
    tracker = WorkflowTracker()
    wf = SimpleWorkflow()

    tracker.add(wf)

    pending = tracker.get_pending()
    assert len(pending) == 1
    assert pending[0] is wf


def test_add_workflow_with_name() -> None:
    tracker = WorkflowTracker()
    wf = SimpleWorkflow()

    tracker.add(wf, name="my_workflow")

    assert tracker.get_name(wf) == "my_workflow"


def test_add_workflow_without_name_returns_none() -> None:
    tracker = WorkflowTracker()
    wf = SimpleWorkflow()

    tracker.add(wf)

    assert tracker.get_name(wf) is None


def test_remove_workflow_from_tracker() -> None:
    tracker = WorkflowTracker()
    wf = SimpleWorkflow()

    tracker.add(wf, name="my_workflow")
    tracker.remove(wf)

    assert tracker.get_pending() == []
    assert tracker.get_name(wf) is None


def test_remove_nonexistent_workflow_is_safe() -> None:
    tracker = WorkflowTracker()
    wf = SimpleWorkflow()

    # Should not raise
    tracker.remove(wf)


def test_add_after_launch_raises_error() -> None:
    tracker = WorkflowTracker()
    tracker.mark_launched()

    wf = SimpleWorkflow()

    with pytest.raises(RuntimeError, match="Cannot add workflows after launch"):
        tracker.add(wf)


def test_is_launched_initially_false() -> None:
    tracker = WorkflowTracker()
    assert tracker.is_launched is False


def test_mark_launched_sets_is_launched() -> None:
    tracker = WorkflowTracker()
    tracker.mark_launched()
    assert tracker.is_launched is True


def test_set_and_get_registered() -> None:
    tracker = WorkflowTracker()
    wf = SimpleWorkflow()

    async def control_loop(
        start_event: object, init_state: object, run_id: str
    ) -> StopEvent:
        return StopEvent(result="done")

    registered = RegisteredWorkflow(
        workflow_function=control_loop,  # type: ignore
        steps={"start": lambda: None},  # type: ignore
    )

    tracker.set_registered(wf, registered)

    assert tracker.get_registered(wf) is registered


def test_get_registered_returns_none_if_not_set() -> None:
    tracker = WorkflowTracker()
    wf = SimpleWorkflow()

    assert tracker.get_registered(wf) is None


def test_clear_resets_all_state() -> None:
    tracker = WorkflowTracker()
    wf = SimpleWorkflow()

    tracker.add(wf, name="my_workflow")
    tracker.mark_launched()

    async def control_loop(
        start_event: object, init_state: object, run_id: str
    ) -> StopEvent:
        return StopEvent(result="done")

    registered = RegisteredWorkflow(
        workflow_function=control_loop,  # type: ignore
        steps={"start": lambda: None},  # type: ignore
    )
    tracker.set_registered(wf, registered)

    tracker.clear()

    assert tracker.get_pending() == []
    assert tracker.get_name(wf) is None
    assert tracker.get_registered(wf) is None
    assert tracker.is_launched is False


def test_strong_refs_survive_gc() -> None:
    """Workflows survive GC when tracked with strong references."""
    tracker = WorkflowTracker()
    wf = SimpleWorkflow()
    w = weakref.ref(wf)

    tracker.add(wf)
    assert len(tracker.get_pending()) == 1

    # Drop external strong reference and force collection
    del wf
    gc.collect()

    # With strong refs, the object should still be alive
    assert w() is not None
    assert len(tracker.get_pending()) == 1


def test_clear_releases_references() -> None:
    """clear() releases strong refs, allowing GC."""
    tracker = WorkflowTracker()
    wf = SimpleWorkflow()
    w = weakref.ref(wf)

    tracker.add(wf, name="my_workflow")
    del wf
    gc.collect()

    # Still alive before clear
    assert w() is not None

    # Clear the tracker
    tracker.clear()
    gc.collect()

    # Now the object should be collected
    assert w() is None


def test_multiple_workflows_tracked_independently() -> None:
    tracker = WorkflowTracker()
    wf1 = SimpleWorkflow()
    wf2 = SimpleWorkflow()

    tracker.add(wf1, name="workflow_1")
    tracker.add(wf2, name="workflow_2")

    assert len(tracker.get_pending()) == 2
    assert tracker.get_name(wf1) == "workflow_1"
    assert tracker.get_name(wf2) == "workflow_2"


def test_add_same_workflow_twice_is_idempotent() -> None:
    """Adding the same workflow twice should not duplicate it."""
    tracker = WorkflowTracker()
    wf = SimpleWorkflow()

    tracker.add(wf, name="my_workflow")
    tracker.add(wf, name="updated_name")  # Second add with different name

    # Should only be one workflow
    assert len(tracker.get_pending()) == 1
    # Name should be updated
    assert tracker.get_name(wf) == "updated_name"
