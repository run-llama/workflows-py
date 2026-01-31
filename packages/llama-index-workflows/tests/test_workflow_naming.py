# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for Workflow.workflow_name property."""

from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent


class SimpleWorkflow(Workflow):
    @step
    async def start(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="done")


def test_explicit_name() -> None:
    """Workflow with explicit name uses that name."""
    wf = SimpleWorkflow(workflow_name="my-custom-name")
    assert wf.workflow_name == "my-custom-name"


def test_default_name_is_module_qualified() -> None:
    """Workflow without name uses module.qualname."""
    wf = SimpleWorkflow()
    # Should be tests.test_workflow_naming.SimpleWorkflow or similar
    assert wf.workflow_name.endswith("SimpleWorkflow")
    assert "." in wf.workflow_name  # Has module prefix


def test_nested_class_qualname() -> None:
    """Nested class workflow has correct qualname."""

    class Outer:
        class InnerWorkflow(Workflow):
            @step
            async def start(self, ev: StartEvent) -> StopEvent:
                return StopEvent(result="done")

    wf = Outer.InnerWorkflow()
    assert "Outer.InnerWorkflow" in wf.workflow_name


def test_function_scoped_workflow_has_locals_in_name() -> None:
    """Function-scoped workflow includes <locals> in name."""

    def create_workflow() -> Workflow:
        class LocalWorkflow(Workflow):
            @step
            async def start(self, ev: StartEvent) -> StopEvent:
                return StopEvent(result="done")

        return LocalWorkflow()

    wf = create_workflow()
    assert "<locals>" in wf.workflow_name
    assert "LocalWorkflow" in wf.workflow_name


def test_explicit_name_overrides_default() -> None:
    """Explicit name takes precedence over computed name."""

    def create_workflow() -> Workflow:
        class LocalWorkflow(Workflow):
            @step
            async def start(self, ev: StartEvent) -> StopEvent:
                return StopEvent(result="done")

        return LocalWorkflow(workflow_name="explicit-name")

    wf = create_workflow()
    assert wf.workflow_name == "explicit-name"
    assert "<locals>" not in wf.workflow_name
