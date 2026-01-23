# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for WorkflowTracker with strong references."""

from __future__ import annotations

import gc
import weakref
from typing import Any, cast

import pytest
from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent
from workflows.runtime.types.plugin import RegisteredWorkflow, WorkflowRunFunction
from workflows.runtime.types.step_function import StepWorkerFunction
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


def test_remove_workflow_from_tracker() -> None:
    tracker = WorkflowTracker()
    wf = SimpleWorkflow()

    tracker.add(wf)
    tracker.remove(wf)

    assert tracker.get_pending() == []


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

    async def mock_workflow_run(
        init_state: object, start_event: object, tags: dict[str, Any]
    ) -> StopEvent:
        return StopEvent(result="done")

    registered = RegisteredWorkflow(
        workflow=wf,
        workflow_run_fn=cast(WorkflowRunFunction, mock_workflow_run),
        steps=cast(dict[str, StepWorkerFunction[Any]], {"start": lambda: None}),
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

    tracker.add(wf)
    tracker.mark_launched()

    async def mock_workflow_run(
        init_state: object, start_event: object, tags: dict[str, Any]
    ) -> StopEvent:
        return StopEvent(result="done")

    registered = RegisteredWorkflow(
        workflow=wf,
        workflow_run_fn=cast(WorkflowRunFunction, mock_workflow_run),
        steps=cast(dict[str, StepWorkerFunction[Any]], {"start": lambda: None}),
    )
    tracker.set_registered(wf, registered)

    tracker.clear()

    assert tracker.get_pending() == []
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

    tracker.add(wf)
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

    tracker.add(wf1)
    tracker.add(wf2)

    assert len(tracker.get_pending()) == 2


def test_add_same_workflow_twice_is_idempotent() -> None:
    """Adding the same workflow twice should not duplicate it."""
    tracker = WorkflowTracker()
    wf = SimpleWorkflow()

    tracker.add(wf)
    tracker.add(wf)  # Second add

    # Should only be one workflow
    assert len(tracker.get_pending()) == 1
