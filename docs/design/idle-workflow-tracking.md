# Idle Workflow Tracking - Design Document

## Overview

This document outlines an approach for tracking when workflows are idle (waiting on external input) in the workflow-server, enabling future release and resumption of idle workflows.

## Problem Statement

Running workflows consume resources (memory, async task slots). When a workflow is waiting for external input via `ctx.wait_for_event()`, it remains in memory even though no computation is occurring. We need a mechanism to:

1. **Detect** when a workflow becomes idle (waiting only on external events)
2. **Track** idle state transitions
3. **Enable** future release of idle workflows while preserving state for resumption

## Design Goals

- **Backend agnostic**: The approach should work with different workflow backends (native, DBOS, Temporal, etc.)
- **Opaque state**: Avoid depending on serialized context structure for idle detection
- **Lightweight**: Minimal overhead, emit events only on state transitions
- **Compatible interface**: Keep events simple to allow future extension without breaking changes

## What Makes a Workflow "Idle"?

A workflow is **idle** when:
1. The workflow is running (hasn't completed/failed/cancelled)
2. All steps have no pending events in their queues
3. All steps have no workers currently executing
4. At least one step has an active waiter (from `ctx.wait_for_event()`)

This state occurs when a step calls `ctx.wait_for_event()` and no other work is available.

## Approach: Internal Events for Idle Transitions

The control loop emits internal events when the workflow transitions to or from an idle state. This keeps idle detection within the runtime where state is already known, and exposes it via the existing event stream.

### New Event Type

```python
# In events.py

class WorkflowIdleEvent(InternalDispatchEvent):
    """Emitted when workflow transitions to idle (waiting on external input)."""
    pass
```

Only one new event is needed. Resumption from idle is already signaled by the existing `StepStateChanged` event with `StepState.RUNNING` - no new event required.

The event is intentionally minimal - no metadata beyond the event type. This maintains a compatible interface and avoids coupling to internal state structure.

### Detection in Control Loop

Idle transition is detected in `_process_step_result_tick`:
- After a step completes and no new work is queued
- Check if all steps now have empty queues and no in-progress work
- If waiters exist and no work remains, emit `WorkflowIdleEvent`

### Implementation

Add a helper function to check idle conditions (computed from existing state, not stored):

```python
def _check_idle_state(state: BrokerState) -> bool:
    """Returns True if workflow is idle (waiting only on external events)."""
    if not state.is_running:
        return False

    for worker_state in state.workers.values():
        if worker_state.queue or worker_state.in_progress:
            return False

    return any(ws.collected_waiters for ws in state.workers.values())
```

Emit idle event in tick processing by comparing before/after state:

```python
def _process_step_result_tick(
    tick: TickStepResult, init: BrokerState, now_seconds: float
) -> tuple[BrokerState, list[WorkflowCommand]]:
    state = init.deepcopy()
    commands: list[WorkflowCommand] = []

    # ... existing step result processing ...

    # Check for idle transition at end of processing
    was_idle = _check_idle_state(init)
    now_idle = _check_idle_state(state)

    if now_idle and not was_idle:
        commands.append(CommandPublishEvent(WorkflowIdleEvent()))

    return state, commands
```

### Server Integration

The server's `_WorkflowHandler` watches for events in `_stream_events`:

```python
@dataclass
class _WorkflowHandler:
    # ... existing fields ...
    idle_since: datetime | None = None

    async def _stream_events(self, on_finish):
        async for event in self.run_handler.stream_events(expose_internal=True):
            # Track idle state
            if isinstance(event, WorkflowIdleEvent):
                self.idle_since = datetime.now(timezone.utc)
            elif isinstance(event, StepStateChanged) and event.step_state == StepState.RUNNING:
                self.idle_since = None  # Resumed from idle

            # ... existing checkpoint and queue logic ...
```

## Serialization Considerations

No changes to workflow context serialization are needed. Idle state is computed from existing state, not stored.

The `idle_since` timestamp is server-level metadata, not part of the workflow context. It should be stored in `PersistentHandler`:

```python
class PersistentHandler(BaseModel):
    # ... existing fields ...
    idle_since: datetime | None = None
```

When a workflow is restored from persistence, the server can recompute idle state on the first checkpoint or when the workflow emits its first event.

## Future Work: Releasing Idle Workflows

With idle tracking in place, future work can implement:

1. **Idle timeout configuration**: How long before an idle workflow is released
2. **Release mechanism**: Cancel the in-memory handler while preserving persisted state
3. **Resume on event**: When an event arrives for a released workflow, restore from persistence and resume
4. **API extensions**: Query idle workflows, manually release/resume

## Edge Cases

1. **Multiple waiters**: Workflow is idle only when ALL work is complete and waiters exist
2. **Queued events**: Not idle if any step has queued events, even with waiters
3. **Retry backoff**: Events in retry backoff are still "queued" - not idle
4. **Rapid transitions**: If workflow quickly becomes idle then resumes, `WorkflowIdleEvent` is still emitted followed by `StepStateChanged(RUNNING)`

## Summary

The event-based approach:
- Emits `WorkflowIdleEvent` when transitioning to idle state
- Resumption detected via existing `StepStateChanged` events (no new event needed)
- Idle state computed from existing `BrokerState` fields (no new stored state)
- Keeps events lightweight with no metadata
- Detection happens in control loop where state is already available
- Server tracks `idle_since` timestamp for duration tracking
- Backend agnostic - any runtime can emit these events
