# Implementation Plan: Idle Workflow Memory Release

## Overview

Release idle workflows from memory after a configurable timeout, and automatically reload them from persistence when an event is received.

## Current State Analysis

### How idle_since works today

1. `_WorkflowHandler.idle_since` is set to `datetime.now(UTC)` when `WorkflowIdleEvent` is emitted (`server.py:1578`)
2. `idle_since` is cleared (set to `None`) when `StepStateChanged(RUNNING)` is emitted (`server.py:1583`)
3. `idle_since` is persisted to the store via `_as_persistent()` (`server.py:1450`)
4. **However**: `idle_since` is NOT restored when resuming from persistence (the wrapper is recreated fresh)

### Memory management today

- `_handlers: dict[str, _WorkflowHandler]` holds all running workflows in memory
- Handlers are only removed when:
  - Workflow completes/fails/cancels (via `on_finish()` callback)
  - Server shuts down (via `stop()`)
  - Explicitly cancelled (via `_cancel_handler()`)
- No TTL-based eviction exists

### Event handling today

- `_post_event` endpoint (`server.py:1067-1187`):
  - If handler in memory: sends event via `ctx.send_event()`
  - If handler NOT in memory but status="running": returns 409 "Handler expired" (line 1141)
  - This is the key gap - we need to reload instead of rejecting

---

## Edge Cases Analysis

### 1. Unhandled event type sent to idle workflow

**Scenario**: Event is sent that doesn't match any step's `accepted_events` and doesn't match any waiter's `waiting_for_event`.

**Current behavior** (`control_loop.py:635-677`):
- Event is checked against all steps' `accepted_events` (line 644)
- Event is checked against all `collected_waiters` (lines 660-676)
- If no match: **silently dropped**, no commands generated

**Impact on memory release**:
- **Safe** - workflow stays idle, `StepStateChanged(RUNNING)` never emitted
- Workflow should be re-released after checking it's still idle
- Need to detect "event processed but workflow still idle" and re-release

### 2. Race condition: event arrives during unload

**Scenario**: Event POST arrives exactly as workflow is being unloaded from memory.

**Mitigation needed**:
- Use a lock/mutex during unload
- If event arrives during unload: complete unload, then reload immediately
- Alternative: abort unload if event pending

### 3. Multiple events arrive for released workflow

**Scenario**: Workflow released, then 3 events arrive rapidly.

**Expected behavior**:
- First event triggers reload
- Subsequent events queue until reload completes
- All events processed in order

### 4. Stream consumers holding references

**Scenario**: Client is actively consuming `stream_events()` for an idle workflow.

**Current behavior**: `consumer_mutex` is held while iterating events.

**Decision needed**:
- Option A: Don't release workflows with active stream consumers
- Option B: Release anyway, consumer gets disconnected
- **Recommendation**: Option A - check `consumer_mutex.locked()` before release

### 5. Workflow becomes idle, then immediately active again

**Scenario**: Brief idle period shorter than release timeout.

**Expected behavior**:
- `idle_since` gets set
- Before timeout expires, step runs, `idle_since` cleared
- No release happens

### 6. Persistence failure during reload

**Scenario**: Event triggers reload, but Context.from_dict() fails.

**Mitigation**:
- Return 500 error to caller
- Log error
- Don't mark workflow as failed (could be transient)

### 7. Workflow released but persistence store unavailable

**Scenario**: Workflow unloaded, then persistence store goes down, then event arrives.

**Behavior**: Reload fails, return 503 to caller (service unavailable)

---

## Implementation Plan

### Phase 1: Core Infrastructure

#### 1.1 Add `HandlerQuery.idle_before` filter

Extend `HandlerQuery` to query by idle time:

```python
@dataclass
class HandlerQuery:
    handler_id_in: List[str] | None = None
    workflow_name_in: List[str] | None = None
    status_in: List[Status] | None = None
    idle_before: datetime | None = None  # NEW: handlers idle before this time
```

Update `MemoryWorkflowStore` and `SqliteWorkflowStore` to support this filter.

#### 1.2 Add server configuration for idle timeout

```python
class WorkflowServer:
    def __init__(
        self,
        ...
        idle_release_timeout: timedelta | None = None,  # None = disabled
    ):
```

#### 1.3 Add `_released_handlers` tracking

Track which handlers have been released so we know to reload them:

```python
self._released_handlers: set[str] = set()
```

Or alternatively, just check: "handler_id in store with status=running but not in _handlers"

### Phase 2: Release Mechanism

#### 2.1 Background task for idle detection

Add periodic task that scans for idle workflows:

```python
async def _idle_release_task(self) -> None:
    """Periodically check for and release idle workflows."""
    while True:
        await asyncio.sleep(self._idle_check_interval)  # e.g., 30 seconds
        await self._release_idle_handlers()

async def _release_idle_handlers(self) -> None:
    """Find and release workflows that have been idle too long."""
    if self._idle_release_timeout is None:
        return

    cutoff = datetime.now(timezone.utc) - self._idle_release_timeout

    for handler_id, wrapper in list(self._handlers.items()):
        if wrapper.idle_since is not None and wrapper.idle_since < cutoff:
            if not wrapper.consumer_mutex.locked():  # No active stream consumers
                await self._release_handler(wrapper)
```

#### 2.2 Release handler method

```python
async def _release_handler(self, wrapper: _WorkflowHandler) -> None:
    """Release an idle handler from memory, keeping it in persistence."""
    handler_id = wrapper.handler_id

    # Final checkpoint to ensure latest state is persisted
    await wrapper.checkpoint()

    # Cancel the streaming task (doesn't cancel the workflow)
    if wrapper.task and not wrapper.task.done():
        wrapper.task.cancel()
        try:
            await wrapper.task
        except asyncio.CancelledError:
            pass

    # Remove from memory
    self._handlers.pop(handler_id, None)
    self._results.pop(handler_id, None)

    logger.info(f"Released idle workflow {handler_id} from memory")
```

**Key insight**: We're not cancelling the workflow itself, just removing the in-memory wrapper. The persisted state retains `status="running"` and can be reloaded.

### Phase 3: Reload Mechanism

#### 3.1 Modify `_post_event` to reload released workflows

Replace the current 409 "Handler expired" response with reload logic:

```python
async def _post_event(self, request: Request) -> JSONResponse:
    handler_id = request.path_params["handler_id"]

    wrapper = self._handlers.get(handler_id)

    # If not in memory, try to reload from persistence
    if wrapper is None:
        wrapper = await self._try_reload_handler(handler_id)
        if wrapper is None:
            raise HTTPException(detail="Handler not found", status_code=404)

    if is_status_completed(wrapper.status):
        raise HTTPException(detail="Workflow already completed", status_code=409)

    # ... rest of event handling ...
```

#### 3.2 Implement reload method

```python
async def _try_reload_handler(self, handler_id: str) -> _WorkflowHandler | None:
    """Attempt to reload a released handler from persistence."""
    handler_data = await self._load_handler(handler_id)

    if handler_data.status != "running":
        return None  # Can't reload completed/failed/cancelled

    workflow = self._workflows.get(handler_data.workflow_name)
    if workflow is None:
        logger.warning(f"Cannot reload {handler_id}: workflow {handler_data.workflow_name} not registered")
        return None

    try:
        context = Context.from_dict(workflow=workflow, data=handler_data.ctx)
        wrapper = await self._start_workflow(
            workflow=_NamedWorkflow(name=handler_data.workflow_name, workflow=workflow),
            handler_id=handler_id,
            context=context,
        )

        # Restore idle_since from persisted state
        wrapper.idle_since = handler_data.idle_since

        logger.info(f"Reloaded workflow {handler_id} from persistence")
        return wrapper
    except Exception as e:
        logger.error(f"Failed to reload handler {handler_id}: {e}")
        raise HTTPException(detail=f"Failed to reload handler: {e}", status_code=500)
```

### Phase 4: Re-release after unhandled events

#### 4.1 Detect "still idle" after event processing

After sending an event, if the workflow remains idle, schedule re-release:

```python
async def _post_event(self, request: Request) -> JSONResponse:
    # ... send event ...

    # Check if workflow went back to idle (event didn't wake it)
    # This happens asynchronously, so we don't check immediately
    # The idle release task will catch it on next scan

    return JSONResponse(SendEventResponse(status="sent").model_dump())
```

The existing idle detection in `_stream_events()` will set `idle_since` again if the workflow becomes idle, and the background task will release it again.

### Phase 5: Stream endpoint reload support

#### 5.1 Modify `_stream_handler_events` to support reload

```python
async def _stream_handler_events(self, request: Request) -> EventSourceResponse:
    handler_id = request.path_params["handler_id"]

    wrapper = self._handlers.get(handler_id)

    # Reload if necessary (for idle workflows receiving stream request)
    if wrapper is None:
        wrapper = await self._try_reload_handler(handler_id)
        if wrapper is None:
            raise HTTPException(detail="Handler not found", status_code=404)

    # ... rest of streaming ...
```

---

## Configuration Options

```python
WorkflowServer(
    idle_release_timeout=timedelta(minutes=5),  # Release after 5 min idle
    idle_check_interval=30.0,  # Check every 30 seconds
)
```

---

## Testing Strategy

1. **Unit tests**:
   - `HandlerQuery.idle_before` filtering
   - Release logic with mock handlers
   - Reload logic from mock persistence

2. **Integration tests**:
   - Workflow becomes idle -> released after timeout
   - Event sent to released workflow -> reloaded
   - Unhandled event -> stays idle -> re-released
   - Stream consumer prevents release
   - Multiple rapid events to released workflow

3. **Edge case tests**:
   - Persistence failure during reload
   - Workflow completes while released
   - Server shutdown with released handlers

---

## Open Questions

1. **Should we expose release/reload in the API?**
   - Add `/handlers/{id}/release` endpoint for manual release?
   - Add `released` to handler status in listings?

2. **Metrics/observability**:
   - Should we emit events when workflows are released/reloaded?
   - Add counters for release/reload operations?

3. **Memory limits**:
   - Should we also release based on total memory pressure?
   - Max handlers in memory setting?

4. **Idle timeout per workflow type**:
   - Should different workflow types have different idle timeouts?
   - E.g., `workflow.register("chat", chat_wf, idle_timeout=timedelta(minutes=10))`
