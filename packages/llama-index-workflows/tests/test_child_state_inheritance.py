"""
Tests for workflow state inheritance behavior.

This module tests the behavior when:
1. A base workflow class uses Context[BaseState]
2. A child workflow class uses Context[ChildState] (where ChildState extends BaseState)

Key findings:
- Multiple state types (BaseState + ChildState) are NOT allowed and raise ValueError
- When all steps use the same state type (e.g., ChildState), inheritance works correctly
- A base class step calling set_state does NOT obliterate child state fields
  (as long as the state type is consistent across all steps)
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


# ============================================================================
# Events for multi-step workflows
# ============================================================================


class MiddleEvent(Event):
    """Event to pass control between steps."""

    pass


# ============================================================================
# Test: Multiple state types raise ValueError
# ============================================================================


class BaseWorkflowMixed(Workflow):
    """Base workflow using Context[BaseState]."""

    @step
    async def start_step(
        self, ctx: Context[BaseState], ev: StartEvent
    ) -> MiddleEvent:
        await ctx.store.set("base_field", "set_by_base")
        return MiddleEvent()


class ChildWorkflowMixed(BaseWorkflowMixed):
    """Child workflow that uses Context[ChildState] - INCOMPATIBLE with base."""

    @step
    async def end_step(self, ctx: Context[ChildState], ev: MiddleEvent) -> StopEvent:
        # This step uses ChildState while the inherited step uses BaseState
        await ctx.store.set("child_field", "set_by_child")
        return StopEvent()


@pytest.mark.asyncio
async def test_mixed_state_types_raises_error() -> None:
    """
    Test that mixing state types (BaseState + ChildState) raises ValueError.

    When a base workflow step uses Context[BaseState] and a child workflow step
    uses Context[ChildState], the Context initialization should fail because
    multiple state types are not supported.
    """
    workflow = ChildWorkflowMixed()
    test_runner = WorkflowTestRunner(workflow)

    with pytest.raises(ValueError) as exc_info:
        await test_runner.run()

    # Verify the error message mentions multiple state types
    assert "Multiple state types are not supported" in str(exc_info.value)
    assert "BaseState" in str(exc_info.value)
    assert "ChildState" in str(exc_info.value)


# ============================================================================
# Test: Using child state type everywhere works correctly
# ============================================================================


class BaseWorkflowConsistent(Workflow):
    """Base workflow using Context[ChildState] (the more specific type)."""

    @step
    async def start_step(
        self, ctx: Context[ChildState], ev: StartEvent
    ) -> MiddleEvent:
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
# Test: set_state does NOT obliterate child fields
# ============================================================================


class BaseWorkflowSetState(Workflow):
    """Base workflow that calls set_state with a complete ChildState."""

    @step
    async def start_step(
        self, ctx: Context[ChildState], ev: StartEvent
    ) -> MiddleEvent:
        # Get current state (ChildState type), modify it, set it back
        state = await ctx.store.get_state()
        state.base_field = "modified_via_set_state"
        # Note: we're setting the entire state object back
        await ctx.store.set_state(state)
        return MiddleEvent()


class ChildWorkflowSetState(BaseWorkflowSetState):
    """Child workflow verifying child fields are preserved after set_state."""

    @step
    async def setup_step(
        self, ctx: Context[ChildState], ev: StartEvent
    ) -> MiddleEvent:
        # First, set up child-specific fields
        await ctx.store.set("child_field", "initial_child_value")
        await ctx.store.set("extra_counter", 42)
        # Now call the inherited start_step by returning MiddleEvent
        # Actually, we need to route through a different event
        return MiddleEvent()


# More explicit test for set_state behavior
class SetStateWorkflow(Workflow):
    """Workflow that tests set_state preserving child fields."""

    @step
    async def init_step(
        self, ctx: Context[ChildState], ev: StartEvent
    ) -> MiddleEvent:
        # Initialize all fields including child-specific ones
        state = await ctx.store.get_state()
        state.base_field = "initial_base"
        state.child_field = "initial_child"
        state.extra_counter = 100
        await ctx.store.set_state(state)
        return MiddleEvent()

    @step
    async def base_like_step(
        self, ctx: Context[ChildState], ev: MiddleEvent
    ) -> StopEvent:
        # This simulates what a base class step might do:
        # get state, modify only base field, set state back
        state = await ctx.store.get_state()
        state.base_field = "modified_base"
        # Critically: we're NOT touching child_field or extra_counter
        await ctx.store.set_state(state)
        return StopEvent()


@pytest.mark.asyncio
async def test_set_state_preserves_child_fields() -> None:
    """
    Test that calling set_state does NOT obliterate child state fields.

    When a step gets the state, modifies only base fields, and calls set_state,
    the child fields should be preserved because:
    1. get_state() returns a copy with all fields
    2. set_state() replaces with the same object (which still has child fields)

    This is the key behavior question: set_state replaces the entire state,
    but if the caller passes back a ChildState object (even if they only
    modified base fields), the child fields remain intact.
    """
    workflow = SetStateWorkflow()
    test_runner = WorkflowTestRunner(workflow)

    result = await test_runner.run()

    ctx = result.ctx
    assert ctx is not None
    state = await ctx.store.get_state()

    # The base field was modified
    assert state.base_field == "modified_base"
    # But child fields should be PRESERVED (not reset to defaults)
    assert state.child_field == "initial_child"
    assert state.extra_counter == 100


# ============================================================================
# Test: set_state type checking prevents setting wrong type
# ============================================================================


@pytest.mark.asyncio
async def test_set_state_wrong_type_raises_error() -> None:
    """
    Test that set_state raises ValueError when setting wrong state type.

    If someone tries to set a BaseState when ChildState is expected,
    it should fail because isinstance(BaseState(), ChildState) is False.
    """
    from workflows.context.state_store import InMemoryStateStore

    # Create a store with ChildState
    store = InMemoryStateStore(ChildState())

    # Try to set a BaseState (parent type) - should fail
    # Intentionally passing wrong type to verify runtime behavior
    with pytest.raises(ValueError) as exc_info:
        await store.set_state(BaseState(base_field="wrong_type"))  # type: ignore[arg-type]

    assert "State must be of type" in str(exc_info.value)


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
    async def init_step(
        self, ctx: Context[ChildState], ev: StartEvent
    ) -> MiddleEvent:
        async with ctx.store.edit_state() as state:
            state.base_field = "initial_base"
            state.child_field = "initial_child"
            state.extra_counter = 50
        return MiddleEvent()

    @step
    async def modify_step(
        self, ctx: Context[ChildState], ev: MiddleEvent
    ) -> StopEvent:
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
