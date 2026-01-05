# Idle Workflow Tracking - Research and Recommendations

## Overview

This document outlines an approach for tracking when workflows are entirely idle (waiting on external input) in the workflow-server, and subsequently releasing them to free resources.

## Problem Statement

Running workflows in the server consume resources (memory, async task slots). When a workflow is waiting for external input via `ctx.wait_for_event()`, it remains in memory even though no computation is occurring. We need a mechanism to:

1. **Detect** when a workflow becomes idle (waiting only on external events)
2. **Track** idle state and duration
3. **Release** idle workflows from memory while preserving their state for later resumption

## Current Architecture

### Workflow Execution Model

Workflows execute via a control loop (`control_loop.py`) that processes "ticks" - discrete state transitions. The `BrokerState` class tracks the complete workflow state:

```python
@dataclass
class BrokerState:
    is_running: bool
    config: BrokerConfig
    workers: dict[str, InternalStepWorkerState]  # Per-step state
```

Each step has its own worker state:

```python
@dataclass
class InternalStepWorkerState:
    queue: list[EventAttempt]           # Events waiting to be processed
    in_progress: list[InProgressState]   # Currently executing workers
    collected_events: dict[str, list[Event]]  # For collect_events()
    collected_waiters: list[StepWorkerWaiter]  # Active wait_for_event() calls
```

### What Makes a Workflow "Idle"?

A workflow is **idle** when:
1. `is_running == True` (workflow hasn't completed)
2. All steps have empty `queue` (no pending events)
3. All steps have empty `in_progress` (no workers executing)
4. At least one step has active `collected_waiters` (waiting for external input)

This state occurs when a step calls `ctx.wait_for_event()` and no other work is available.

### Current State Tracking

The server already tracks handler state via `PersistentHandler`:

```python
class PersistentHandler(BaseModel):
    handler_id: str
    workflow_name: str
    status: Status  # "running" | "completed" | "failed" | "cancelled"
    updated_at: datetime | None
    ctx: dict[str, Any]  # Serialized workflow context
```

**Key insight**: The `ctx` field contains the full serialized `BrokerState`, which already includes all information needed to detect idleness.

## Recommended Approach

### Option 1: Add Idle Detection Method to BrokerState (Recommended)

Add a method to `BrokerState` that determines if the workflow is idle:

```python
# In runtime/types/internal_state.py

@dataclass
class BrokerState:
    # ... existing fields ...

    def is_idle(self) -> bool:
        """Returns True if workflow is waiting only on external events."""
        if not self.is_running:
            return False

        has_waiters = False
        for worker_state in self.workers.values():
            # If any step has work to do, not idle
            if worker_state.queue or worker_state.in_progress:
                return False
            if worker_state.collected_waiters:
                has_waiters = True

        # Idle if running with no work but has waiters
        return has_waiters

    def get_active_waiters(self) -> list[tuple[str, StepWorkerWaiter]]:
        """Returns list of (step_name, waiter) for all active waiters."""
        waiters = []
        for step_name, worker_state in self.workers.items():
            for waiter in worker_state.collected_waiters:
                waiters.append((step_name, waiter))
        return waiters
```

**Pros:**
- Clean separation of concerns - idle detection is a property of workflow state
- Reusable across different contexts (server, testing, debugging)
- Can be computed from serialized state without running workflow

**Cons:**
- Requires state snapshot to be current

### Option 2: Track Idle State in Server Handler Wrapper

Track idle state in `_WorkflowHandler` via event stream monitoring:

```python
# In server/server.py

@dataclass
class _WorkflowHandler:
    # ... existing fields ...
    idle_since: datetime | None = None

    async def _stream_events(self, on_finish):
        # ... existing code ...
        async for event in self.run_handler.stream_events(expose_internal=True):
            if isinstance(event, StepStateChanged):
                if event.step_state == StepState.NOT_RUNNING:
                    # Check if workflow is now idle
                    if self._check_idle():
                        self.idle_since = datetime.now(timezone.utc)
                elif event.step_state == StepState.RUNNING:
                    self.idle_since = None  # No longer idle
            # ... rest of streaming logic ...
```

**Pros:**
- Real-time tracking without polling
- Natural integration with existing event stream

**Cons:**
- Requires access to `BrokerState` to check idleness
- More complex integration

### Option 3: Event-Based Idle Notification (Comprehensive)

Emit a dedicated `WorkflowIdleEvent` from the control loop when the workflow becomes idle:

```python
# In events.py
class WorkflowIdleEvent(InternalDispatchEvent):
    """Emitted when workflow has no work except waiting on external events."""
    waiters: list[dict]  # Info about what we're waiting for
    idle_since: float    # Timestamp when idleness began

# In control_loop.py, after processing each tick:
if state.is_idle() and not previous_state.is_idle():
    commands.append(CommandPublishEvent(
        WorkflowIdleEvent(
            waiters=[...],
            idle_since=now_seconds
        )
    ))
```

**Pros:**
- Explicit notification - no polling needed
- Carries metadata about what workflow is waiting for
- Can trigger immediate action in server

**Cons:**
- Requires changes to control loop
- Additional event type to handle

## Recommended Implementation Plan

### Phase 1: Add Idle Detection to BrokerState

1. Add `is_idle()` method to `BrokerState` in `runtime/types/internal_state.py`
2. Add `get_active_waiters()` method for debugging/logging
3. Add unit tests for idle detection logic

### Phase 2: Track Idle State in Server

1. Add `idle_since: datetime | None` field to `_WorkflowHandler`
2. Update `_stream_events()` to track when workflow becomes idle
3. Add idle state to `PersistentHandler` and checkpoint it
4. Add `idle_since` to handler API responses

### Phase 3: Idle Workflow Release (Future)

1. Add configuration for idle timeout
2. Implement background task to check for idle workflows
3. Release idle workflows from memory (cancel handler, keep persisted state)
4. Implement resumption when events arrive for released workflows

## Implementation Details

### Detecting Idle State Transitions

The key is to detect transitions **to** and **from** idle state:

```python
# Pseudo-code for idle transition detection
async def _stream_events(self, on_finish):
    was_idle = False

    async for event in self.run_handler.stream_events(expose_internal=True):
        if isinstance(event, StepStateChanged):
            # After step completes, check if now idle
            broker_state = self._get_current_broker_state()
            is_idle = broker_state.is_idle()

            if is_idle and not was_idle:
                # Transition TO idle
                self.idle_since = datetime.now(timezone.utc)
                logger.info(f"Handler {self.handler_id} became idle")
            elif not is_idle and was_idle:
                # Transition FROM idle (work resumed)
                self.idle_since = None
                logger.info(f"Handler {self.handler_id} resumed")

            was_idle = is_idle
```

### Accessing BrokerState from Server

The challenge is that `BrokerState` lives inside the running workflow, not directly accessible from the server. Options:

1. **Via Context serialization**: `ctx.to_dict()` includes broker state
2. **Add property to WorkflowHandler**: Expose `broker_state` property
3. **Via dedicated event**: Emit idle events from control loop

**Recommended**: Add a property to `Context` that exposes idle state:

```python
# In context/context.py
class Context:
    @property
    def is_idle(self) -> bool:
        """Returns True if workflow is waiting only on external events."""
        return self._broker._state.is_idle() if self._broker else False
```

### Persisting Idle State

Extend `PersistentHandler` to track idle information:

```python
class PersistentHandler(BaseModel):
    # ... existing fields ...
    idle_since: datetime | None = None
    active_waiters: list[dict] | None = None  # Optional metadata
```

This allows:
- Querying for idle workflows: `status="running" AND idle_since IS NOT NULL`
- Computing idle duration: `now - idle_since`
- Knowing what workflows are waiting for

## API Changes

### Handler Response

Add idle information to handler API responses:

```python
class HandlerData(BaseModel):
    # ... existing fields ...
    idle_since: str | None = None  # ISO timestamp when became idle
    active_waiters: list[dict] | None = None  # What we're waiting for
```

### Query API

Add ability to filter by idle state:

```
GET /handlers?status=running&idle=true
GET /handlers?idle_duration_gt=300  # Idle for more than 5 minutes
```

## Considerations

### Edge Cases

1. **Workflow with multiple waiters**: Idle when ALL waiters are waiting
2. **Workflow with queued events**: Not idle even if waiters exist
3. **Workflow in retry backoff**: Consider if this counts as "work in progress"
4. **Nested workflows**: Each tracked independently

### Performance

- Idle detection is O(n) where n = number of steps (typically small)
- Should be computed on-demand or on state transitions, not polled
- Consider caching idle state to avoid repeated computation

### Serialization

The `is_idle` state is derived from existing serialized state, so no schema changes needed for context serialization. The `idle_since` timestamp would be new metadata to track.

## Summary

**Recommended approach**:

1. Add `is_idle()` method to `BrokerState` for clean detection logic
2. Track `idle_since` timestamp in `_WorkflowHandler` by monitoring step state changes
3. Persist idle metadata in `PersistentHandler` for querying
4. Expose idle state in handler API responses

This provides a foundation for future work on releasing idle workflows while maintaining full state for resumption.
