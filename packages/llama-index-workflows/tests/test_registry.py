# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

"""Tests for the workflow registry module."""

from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent
from workflows.plugins import WorkflowRegistry
from workflows.plugins._registry import _WorkflowRegistry, compute_workflow_id

# --- compute_workflow_id tests ---


def test_compute_workflow_id_class_format() -> None:
    class SimpleWorkflow(Workflow):
        @step
        async def do_work(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

    wf_id = compute_workflow_id(SimpleWorkflow)
    assert wf_id.startswith("SimpleWorkflow:")
    assert len(wf_id) == len("SimpleWorkflow:") + 12  # 12 char hash


def test_compute_workflow_id_same_steps_same_hash() -> None:
    class WorkflowA(Workflow):
        @step
        async def step1(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

    class WorkflowB(Workflow):
        @step
        async def step1(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

    id_a = compute_workflow_id(WorkflowA)
    id_b = compute_workflow_id(WorkflowB)

    # Same step names -> same hash portion
    assert id_a.split(":")[1] == id_b.split(":")[1]
    # But different class names
    assert id_a != id_b


def test_compute_workflow_id_different_steps_different_hash() -> None:
    class WorkflowC(Workflow):
        @step
        async def alpha(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

    class WorkflowD(Workflow):
        @step
        async def beta(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

    id_c = compute_workflow_id(WorkflowC)
    id_d = compute_workflow_id(WorkflowD)

    # Different step names -> different hashes
    assert id_c.split(":")[1] != id_d.split(":")[1]


def test_compute_workflow_id_instance_matches_class() -> None:
    class MyWorkflow(Workflow):
        @step
        async def do_work(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

    instance = MyWorkflow()
    assert compute_workflow_id(instance) == compute_workflow_id(MyWorkflow)


def test_compute_workflow_id_consistent_across_calls() -> None:
    class ConsistentWorkflow(Workflow):
        @step
        async def process(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

    id1 = compute_workflow_id(ConsistentWorkflow)
    id2 = compute_workflow_id(ConsistentWorkflow)
    id3 = compute_workflow_id(ConsistentWorkflow)

    assert id1 == id2 == id3


def test_compute_workflow_id_multiple_steps_sorted() -> None:
    """Steps should be sorted, so order of definition doesn't matter for the hash."""

    class WorkflowWithManySteps(Workflow):
        @step
        async def zebra(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

        @step
        async def alpha(self, ev: StopEvent) -> None:
            pass

    # The hash should be computed from sorted step names ["alpha", "zebra"]
    wf_id = compute_workflow_id(WorkflowWithManySteps)
    assert wf_id.startswith("WorkflowWithManySteps:")


# --- WorkflowRegistry tests ---


def test_registry_discovers_workflow_subclasses() -> None:
    """Workflow subclasses are automatically discovered via __subclasses__()."""

    class DiscoveredWorkflow(Workflow):
        @step
        async def do_work(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

    assert DiscoveredWorkflow in WorkflowRegistry.classes
    assert WorkflowRegistry.get_class_id(DiscoveredWorkflow) is not None


def test_registry_base_workflow_not_registered() -> None:
    """The base Workflow class should NOT be in the registry."""
    assert Workflow not in WorkflowRegistry.classes


def test_registry_class_id_format() -> None:
    """get_class_id returns the expected ID format."""

    class FormatTest(Workflow):
        @step
        async def do_work(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

    class_id = WorkflowRegistry.get_class_id(FormatTest)
    assert class_id is not None
    assert class_id.startswith("FormatTest:")
    assert len(class_id.split(":")[1]) == 12


def test_registry_instance_registration() -> None:
    """Instances can be explicitly registered with a name."""

    class InstanceTest(Workflow):
        @step
        async def do_work(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

    instance = InstanceTest()
    wf_id = WorkflowRegistry.register_instance(instance, name="my-instance")

    assert wf_id == "my-instance"
    assert WorkflowRegistry.get_instance_id(instance) == "my-instance"


def test_registry_instance_registration_without_name() -> None:
    """Instances registered without a name use their computed ID."""

    class NoNameTest(Workflow):
        @step
        async def do_work(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

    instance = NoNameTest()
    wf_id = WorkflowRegistry.register_instance(instance)

    # Should use computed ID
    assert wf_id == compute_workflow_id(instance)


def test_registry_iter_classes() -> None:
    """iter_classes yields (class, id) pairs."""

    class IterTest(Workflow):
        @step
        async def do_work(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

    found = False
    for cls, wf_id in WorkflowRegistry.iter_classes():
        if cls is IterTest:
            assert wf_id == compute_workflow_id(IterTest)
            found = True
            break
    assert found


def test_registry_iter_instances() -> None:
    """iter_instances yields (instance, name, computed_id) tuples."""

    class IterInstanceTest(Workflow):
        @step
        async def do_work(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

    instance = IterInstanceTest()
    WorkflowRegistry.register_instance(instance, name="test-iter-instance")

    found = False
    for inst, name, computed_id in WorkflowRegistry.iter_instances():
        if inst is instance:
            assert name == "test-iter-instance"
            assert computed_id == compute_workflow_id(instance)
            found = True
            break
    assert found


def test_registry_fresh_registry_has_no_instances() -> None:
    """A fresh WorkflowRegistry has no registered instances."""
    fresh = _WorkflowRegistry()
    # Classes are discovered via __subclasses__(), so fresh registries still see them
    # But instances must be explicitly registered
    assert len(fresh.instances) == 0


def test_registry_get_instance_computed_id() -> None:
    """get_instance_computed_id returns the hash-based ID."""

    class ComputedIdTest(Workflow):
        @step
        async def do_work(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

    instance = ComputedIdTest()
    WorkflowRegistry.register_instance(instance, name="custom-name")

    # get_instance_id returns the custom name
    assert WorkflowRegistry.get_instance_id(instance) == "custom-name"
    # get_instance_computed_id returns the hash-based ID
    computed = WorkflowRegistry.get_instance_computed_id(instance)
    assert computed is not None
    assert computed.startswith("ComputedIdTest:")


# --- is_modified tests ---


def test_workflow_is_modified_false_for_method_only_workflow() -> None:
    """Workflows with only method-defined steps are not modified."""

    class MethodOnlyWorkflow(Workflow):
        @step
        async def do_work(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

    instance = MethodOnlyWorkflow()
    assert instance.is_modified is False


def test_workflow_is_modified_true_after_add_step() -> None:
    """Workflows become modified after add_step() is called."""

    class DynamicWorkflow(Workflow):
        @step
        async def do_work(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

    # Before adding a step - not modified
    assert DynamicWorkflow().is_modified is False

    # Add a dynamic step using the workflow parameter in the decorator
    @step(workflow=DynamicWorkflow)
    async def extra_step(ev: StopEvent) -> None:
        pass

    # After adding a step - modified
    assert DynamicWorkflow().is_modified is True


def test_registry_instance_registration_unhashable_workflow() -> None:
    """Instances with unhashable types (e.g., custom __eq__ without __hash__) can be registered."""

    class UnhashableWorkflow(Workflow):
        """A workflow that defines __eq__ without __hash__, making instances unhashable."""

        def __eq__(self, other: object) -> bool:
            return self is other

        @step
        async def do_work(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

    instance = UnhashableWorkflow()

    # Verify the instance is actually unhashable
    try:
        hash(instance)
        raise AssertionError("Expected instance to be unhashable")
    except TypeError:
        pass  # Expected

    # Registration should still work via identity-based hashing
    wf_id = WorkflowRegistry.register_instance(instance, name="unhashable-test")
    assert wf_id == "unhashable-test"
    assert WorkflowRegistry.get_instance_id(instance) == "unhashable-test"
