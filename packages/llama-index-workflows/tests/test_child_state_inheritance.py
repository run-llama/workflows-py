"""
Tests for workflow state inheritance behavior.

This module tests the behavior when:
1. A base workflow class uses Context[BaseState]
2. A child workflow class uses Context[ChildState] (where ChildState extends BaseState)

Key behavior:
- Subtype relationships are allowed (BaseState + ChildState work together)
- The most derived type (ChildState) is used as the actual state type
- When a base class step calls set_state with a BaseState, the child fields
  are preserved through merging (not obliterated)
"""

import pytest
from pydantic import BaseModel, Field
from workflows import Context, Workflow
from workflows.decorators import step
from workflows.events import Event, StartEvent, StopEvent
from workflows.testing import WorkflowTestRunner

# ============================================================================
# State models for testing inheritance
# ============================================================================


class BaseState(BaseModel):
    """Base state with a single field."""

    base_field: str = Field(default="base_default")


class ChildState(BaseState):
    """Child state that extends BaseState with additional fields."""

    child_field: str = Field(default="child_default")
    extra_counter: int = Field(default=0)


class UnrelatedState(BaseModel):
    """State that is NOT in the BaseState/ChildState hierarchy."""

    unrelated_field: str = Field(default="unrelated")


# ============================================================================
# Events for multi-step workflows
# ============================================================================


class MiddleEvent(Event):
    """Event to pass control between steps."""

    pass


# ============================================================================
# Test: Subtype state inheritance works correctly
# ============================================================================


class BaseWorkflowWithBaseState(Workflow):
    """Base workflow using Context[BaseState]."""

    @step
    async def base_step(self, ctx: Context[BaseState], ev: StartEvent) -> MiddleEvent:
        # Base step works with BaseState type, sets base field
        await ctx.store.set("base_field", "set_by_base_step")
        return MiddleEvent()


class ChildWorkflowWithChildState(BaseWorkflowWithBaseState):
    """Child workflow that uses Context[ChildState] - now compatible with base."""

    @step
    async def child_step(self, ctx: Context[ChildState], ev: MiddleEvent) -> StopEvent:
        # Child step can access both base and child fields
        await ctx.store.set("child_field", "set_by_child_step")
        return StopEvent()


@pytest.mark.asyncio
async def test_subtype_inheritance_works() -> None:
    """
    Test that subtype state inheritance works correctly.

    When a base workflow step uses Context[BaseState] and a child workflow step
    uses Context[ChildState], the system should:
    1. Use ChildState (most derived) as the actual state type
    2. Allow both steps to work with the state
    3. Preserve all fields from both base and child
    """
    workflow = ChildWorkflowWithChildState()
    test_runner = WorkflowTestRunner(workflow)

    result = await test_runner.run()

    ctx = result.ctx
    assert ctx is not None
    state = await ctx.store.get_state()

    # Verify state is ChildState
    assert isinstance(state, ChildState)
    # Both base and child fields should be properly set
    assert state.base_field == "set_by_base_step"
    assert state.child_field == "set_by_child_step"


# ============================================================================
# Test: set_state with parent type merges fields (doesn't obliterate)
# ============================================================================


class WorkflowWithBaseStateSetState(Workflow):
    """Workflow where base step calls set_state with BaseState."""

    @step
    async def init_step(self, ctx: Context[ChildState], ev: StartEvent) -> MiddleEvent:
        # Initialize all fields including child-specific ones
        await ctx.store.set("base_field", "initial_base")
        await ctx.store.set("child_field", "initial_child")
        await ctx.store.set("extra_counter", 100)
        return MiddleEvent()

    @step
    async def base_step(self, ctx: Context[BaseState], ev: MiddleEvent) -> StopEvent:
        # This step only knows about BaseState, creates a new BaseState
        # and sets it. This should merge, not obliterate child fields.
        new_state = BaseState(base_field="modified_by_base_step")
        await ctx.store.set_state(new_state)  # type: ignore[arg-type]
        return StopEvent()


@pytest.mark.asyncio
async def test_set_state_with_parent_type_merges_fields() -> None:
    """
    Test that set_state with a parent type merges fields, not obliterates.

    When a base class step creates a new BaseState and calls set_state,
    the child fields (child_field, extra_counter) should be preserved
    while the base field is updated.
    """
    workflow = WorkflowWithBaseStateSetState()
    test_runner = WorkflowTestRunner(workflow)

    result = await test_runner.run()

    ctx = result.ctx
    assert ctx is not None
    state = await ctx.store.get_state()

    # The base field was modified
    assert state.base_field == "modified_by_base_step"
    # Child fields should be PRESERVED (not reset to defaults)
    assert state.child_field == "initial_child"
    assert state.extra_counter == 100


# ============================================================================
# Test: Incompatible state types still raise error
# ============================================================================


class WorkflowWithUnrelatedState(Workflow):
    """Workflow with an unrelated state type."""

    @step
    async def step_one(self, ctx: Context[BaseState], ev: StartEvent) -> MiddleEvent:
        return MiddleEvent()

    @step
    async def step_two(
        self, ctx: Context[UnrelatedState], ev: MiddleEvent
    ) -> StopEvent:
        return StopEvent()


@pytest.mark.asyncio
async def test_incompatible_state_types_raises_error() -> None:
    """
    Test that incompatible state types (not in same hierarchy) raise ValueError.

    When state types are not in a parent-child relationship, they are
    incompatible and should raise an error.
    """
    workflow = WorkflowWithUnrelatedState()
    test_runner = WorkflowTestRunner(workflow)

    with pytest.raises(ValueError) as exc_info:
        await test_runner.run()

    # Verify the error message mentions incompatible hierarchy
    assert "not in a compatible inheritance hierarchy" in str(exc_info.value)


# ============================================================================
# Test: Using child state type everywhere still works
# ============================================================================


class BaseWorkflowConsistent(Workflow):
    """Base workflow using Context[ChildState] (the more specific type)."""

    @step
    async def start_step(self, ctx: Context[ChildState], ev: StartEvent) -> MiddleEvent:
        # Base class step modifies state
        await ctx.store.set("base_field", "modified_by_base_step")
        return MiddleEvent()


class ChildWorkflowConsistent(BaseWorkflowConsistent):
    """Child workflow that also uses Context[ChildState] - COMPATIBLE."""

    @step
    async def end_step(self, ctx: Context[ChildState], ev: MiddleEvent) -> StopEvent:
        # Child step can access and modify the same state
        state = await ctx.store.get_state()
        await ctx.store.set("child_field", "modified_by_child_step")
        await ctx.store.set("extra_counter", state.extra_counter + 1)
        return StopEvent()


@pytest.mark.asyncio
async def test_consistent_child_state_works() -> None:
    """
    Test that using the same child state type in both base and child works.

    When all steps (inherited and new) use the same state type, the workflow
    should execute without errors and both base and child fields should be
    accessible.
    """
    workflow = ChildWorkflowConsistent()
    test_runner = WorkflowTestRunner(workflow)

    result = await test_runner.run()

    ctx = result.ctx
    assert ctx is not None
    state = await ctx.store.get_state()

    # Both base and child fields should be properly set
    assert state.base_field == "modified_by_base_step"
    assert state.child_field == "modified_by_child_step"
    assert state.extra_counter == 1


# ============================================================================
# Test: set_state with same type works (direct replacement)
# ============================================================================


class SetStateWorkflow(Workflow):
    """Workflow that tests set_state with same type."""

    @step
    async def init_step(self, ctx: Context[ChildState], ev: StartEvent) -> MiddleEvent:
        # Initialize all fields including child-specific ones
        state = await ctx.store.get_state()
        state.base_field = "initial_base"
        state.child_field = "initial_child"
        state.extra_counter = 100
        await ctx.store.set_state(state)
        return MiddleEvent()

    @step
    async def modify_step(self, ctx: Context[ChildState], ev: MiddleEvent) -> StopEvent:
        # Get state, modify, and set back (same type)
        state = await ctx.store.get_state()
        state.base_field = "modified_base"
        await ctx.store.set_state(state)
        return StopEvent()


@pytest.mark.asyncio
async def test_set_state_same_type_preserves_fields() -> None:
    """
    Test that calling set_state with same type preserves all fields.

    When a step gets the state, modifies only base fields, and calls set_state,
    the child fields should be preserved because it's the same ChildState object.
    """
    workflow = SetStateWorkflow()
    test_runner = WorkflowTestRunner(workflow)

    result = await test_runner.run()

    ctx = result.ctx
    assert ctx is not None
    state = await ctx.store.get_state()

    # The base field was modified
    assert state.base_field == "modified_base"
    # But child fields should be PRESERVED
    assert state.child_field == "initial_child"
    assert state.extra_counter == 100


# ============================================================================
# Test: set_state with unrelated type raises error
# ============================================================================


@pytest.mark.asyncio
async def test_set_state_unrelated_type_raises_error() -> None:
    """
    Test that set_state raises ValueError when setting unrelated state type.

    If someone tries to set an UnrelatedState when ChildState is expected,
    it should fail because they are not in the same inheritance hierarchy.
    """
    from workflows.context.state_store import InMemoryStateStore

    # Create a store with ChildState
    store = InMemoryStateStore(ChildState())

    # Try to set an UnrelatedState - should fail
    with pytest.raises(ValueError) as exc_info:
        await store.set_state(UnrelatedState(unrelated_field="test"))  # type: ignore[arg-type]

    assert "must be of type" in str(exc_info.value)


# ============================================================================
# Test: set_state with parent type at store level
# ============================================================================


@pytest.mark.asyncio
async def test_set_state_parent_type_merges_at_store_level() -> None:
    """
    Test that set_state with parent type merges fields at store level.

    Directly test the InMemoryStateStore behavior when setting a parent
    type onto a child state.
    """
    from workflows.context.state_store import InMemoryStateStore

    # Create a store with ChildState and set some initial values
    initial_state = ChildState(
        base_field="initial_base", child_field="initial_child", extra_counter=42
    )
    store = InMemoryStateStore(initial_state)

    # Set a BaseState (parent type) - should merge, not replace
    new_base_state = BaseState(base_field="updated_base")
    await store.set_state(new_base_state)  # type: ignore[arg-type]

    # Verify merging behavior
    result = await store.get_state()
    assert isinstance(result, ChildState)
    assert result.base_field == "updated_base"  # Updated from parent
    assert result.child_field == "initial_child"  # Preserved
    assert result.extra_counter == 42  # Preserved


# ============================================================================
# Test: Using DictState as a flexible alternative
# ============================================================================


class BaseWorkflowDictState(Workflow):
    """Base workflow using untyped Context (DictState)."""

    @step
    async def start_step(self, ctx: Context, ev: StartEvent) -> MiddleEvent:
        await ctx.store.set("base_field", "set_by_base")
        return MiddleEvent()


class ChildWorkflowDictState(BaseWorkflowDictState):
    """Child workflow that also uses untyped Context."""

    @step
    async def end_step(self, ctx: Context, ev: MiddleEvent) -> StopEvent:
        await ctx.store.set("child_field", "set_by_child")
        return StopEvent()


@pytest.mark.asyncio
async def test_dict_state_allows_flexible_inheritance() -> None:
    """
    Test that using DictState (untyped Context) allows flexible inheritance.

    When workflows don't specify a state type, DictState is used which allows
    any fields to be set dynamically. This is a valid pattern for inheritance
    when type safety is not required.
    """
    workflow = ChildWorkflowDictState()
    test_runner = WorkflowTestRunner(workflow)

    result = await test_runner.run()

    ctx = result.ctx
    assert ctx is not None

    # Both fields can be retrieved
    base_field = await ctx.store.get("base_field")
    child_field = await ctx.store.get("child_field")

    assert base_field == "set_by_base"
    assert child_field == "set_by_child"


# ============================================================================
# Test: edit_state context manager also preserves fields
# ============================================================================


class EditStateWorkflow(Workflow):
    """Workflow testing edit_state preserves child fields."""

    @step
    async def init_step(self, ctx: Context[ChildState], ev: StartEvent) -> MiddleEvent:
        async with ctx.store.edit_state() as state:
            state.base_field = "initial_base"
            state.child_field = "initial_child"
            state.extra_counter = 50
        return MiddleEvent()

    @step
    async def modify_step(self, ctx: Context[ChildState], ev: MiddleEvent) -> StopEvent:
        # Use edit_state to modify only base field
        async with ctx.store.edit_state() as state:
            state.base_field = "edited_base"
            # Not touching child fields
        return StopEvent()


@pytest.mark.asyncio
async def test_edit_state_preserves_child_fields() -> None:
    """
    Test that edit_state context manager preserves unmodified child fields.

    When using the edit_state context manager, only the fields that are
    explicitly modified should change; other fields remain intact.
    """
    workflow = EditStateWorkflow()
    test_runner = WorkflowTestRunner(workflow)

    result = await test_runner.run()

    ctx = result.ctx
    assert ctx is not None
    state = await ctx.store.get_state()

    # Base field was modified
    assert state.base_field == "edited_base"
    # Child fields should be preserved
    assert state.child_field == "initial_child"
    assert state.extra_counter == 50


# ============================================================================
# Test: Three-level inheritance hierarchy
# ============================================================================


class GrandchildState(ChildState):
    """Grandchild state with an additional field."""

    grandchild_field: str = Field(default="grandchild_default")


class BaseWorkflowThreeLevel(Workflow):
    """Base workflow using BaseState."""

    @step
    async def level1_step(self, ctx: Context[BaseState], ev: StartEvent) -> MiddleEvent:
        await ctx.store.set("base_field", "set_at_level1")
        return MiddleEvent()


class ChildWorkflowThreeLevel(BaseWorkflowThreeLevel):
    """Middle-level workflow using ChildState."""

    @step
    async def level2_step(
        self, ctx: Context[ChildState], ev: MiddleEvent
    ) -> MiddleEvent:
        await ctx.store.set("child_field", "set_at_level2")
        return MiddleEvent()


class GrandchildWorkflowThreeLevel(ChildWorkflowThreeLevel):
    """Leaf workflow using GrandchildState."""

    @step
    async def level3_step(
        self, ctx: Context[GrandchildState], ev: MiddleEvent
    ) -> StopEvent:
        await ctx.store.set("grandchild_field", "set_at_level3")
        return StopEvent()


@pytest.mark.asyncio
async def test_three_level_inheritance_works() -> None:
    """
    Test that three-level state inheritance works correctly.

    When workflows have a three-level inheritance hierarchy
    (BaseState -> ChildState -> GrandchildState), the most derived type
    should be used and all fields should be accessible.
    """
    workflow = GrandchildWorkflowThreeLevel()
    test_runner = WorkflowTestRunner(workflow)

    result = await test_runner.run()

    ctx = result.ctx
    assert ctx is not None
    state = await ctx.store.get_state()

    # Verify state is GrandchildState
    assert isinstance(state, GrandchildState)
    # All fields from all levels should be properly set
    assert state.base_field == "set_at_level1"
    assert state.child_field == "set_at_level2"
    assert state.grandchild_field == "set_at_level3"
