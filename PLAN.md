# Implementation Plan: Release Idle Workflows from Memory

## Overview

This plan covers releasing idle workflows from memory and only resuming/reloading them when an input signal is later sent. This is a memory optimization feature for long-running workflow servers that may have many workflows waiting for external input.

## Current State Analysis

### Key Components
- **`WorkflowServer._handlers: dict[str, _WorkflowHandler]`** - In-memory registry of active handlers (`server.py:81`)
- **`_WorkflowHandler`** - Wrapper holding runtime handler, event queue, and streaming task (`server.py:1410-1650`)
- **`PersistentHandler`** - Serializable model for persistence (`abstract_workflow_store.py:30-64`)
- **`AbstractWorkflowStore`** - Interface for persistence (memory or SQLite)

### Current Workflow Lifecycle
1. **Start**: Workflow created, registered in `_handlers`, persisted to store with status `"running"`
2. **Running**: Events streamed, checkpoints on each `StepStateChanged(NOT_RUNNING)`
3. **Idle**: Workflow emits `InputRequiredEvent` or calls `wait_for_event()`, stays in memory waiting
4. **Resume on Input**: Event sent via `POST /events/{handler_id}`, dispatched to context
5. **Complete**: Handler removed from `_handlers`, final checkpoint with terminal status

### Problem
Idle workflows remain in `_handlers` consuming memory indefinitely while waiting for external input.

---

## Implementation Plan

### Phase 1: Add `idle_since` Tracking

**Files to modify:**
- `abstract_workflow_store.py`
- `server.py`
- SQLite migration (new file)

**Changes:**

1. **Add `idle_since` to `PersistentHandler`** (`abstract_workflow_store.py:30-64`)
   ```python
   class PersistentHandler(BaseModel):
       # ... existing fields ...
       idle_since: datetime | None = None  # When workflow last became idle
   ```

2. **Add `idle_since` to `_WorkflowHandler`** (`server.py:1410-1430`)
   ```python
   @dataclass
   class _WorkflowHandler:
       # ... existing fields ...
       idle_since: datetime | None = None
   ```

3. **Track idle state in `_stream_events()`** (`server.py:1567-1600`)
   - Detect when `InputRequiredEvent` or similar "waiting" events are emitted
   - Set `idle_since = datetime.now(timezone.utc)` when workflow becomes idle
   - Clear `idle_since = None` when workflow receives an event and resumes processing

4. **Create SQLite migration** (`migrations/0003_add_idle_since.sql`)
   ```sql
   ALTER TABLE handlers ADD COLUMN idle_since TEXT;
   ```

---

### Phase 2: Implement Workflow Release Mechanism

**New concept: "Released" vs "In-Memory"**
- Released workflows have their runtime handler closed but remain persisted with `status="running"` and `idle_since` set
- They can be reloaded on-demand when an event arrives

**Files to modify:**
- `server.py`
- `abstract_workflow_store.py` (optional: add query by idle_since)

**Changes:**

1. **Add `release_idle_workflow()` method to `WorkflowServer`**
   ```python
   async def release_idle_workflow(self, handler_id: str) -> bool:
       """
       Release an idle workflow from memory while keeping it persisted.
       Returns True if released, False if not found or not idle.
       """
       wrapper = self._handlers.get(handler_id)
       if wrapper is None or wrapper.idle_since is None:
           return False

       # Checkpoint current state before release
       await wrapper.checkpoint()

       # Cancel the streaming task (but not the workflow itself)
       if wrapper.task is not None:
           wrapper.task.cancel()
           try:
               await wrapper.task
           except asyncio.CancelledError:
               pass

       # Remove from in-memory handlers
       self._handlers.pop(handler_id, None)
       return True
   ```

2. **Add `_reload_workflow()` method to `WorkflowServer`**
   ```python
   async def _reload_workflow(self, handler_id: str) -> _WorkflowHandler | None:
       """
       Reload a released workflow from persistence back into memory.
       Returns the handler if successful, None if not found.
       """
       persisted = await self._workflow_store.query(
           HandlerQuery(handler_id_in=[handler_id], status_in=["running"])
       )
       if not persisted:
           return None

       persistent = persisted[0]
       workflow = self._workflows.get(persistent.workflow_name)
       if workflow is None:
           return None

       # Resume the workflow from persisted context
       context = Context.from_dict(workflow=workflow, data=persistent.ctx)
       wrapper = await self._start_workflow(
           workflow=_NamedWorkflow(name=persistent.workflow_name, workflow=workflow),
           handler_id=handler_id,
           context=context,
       )
       return wrapper
   ```

---

### Phase 3: Modify Event Posting to Auto-Reload

**File to modify:** `server.py` (`_post_event` method, lines 1066-1186)

**Current behavior** (`server.py:1127-1140`):
```python
wrapper = self._handlers.get(handler_id)
if wrapper is not None and is_status_completed(wrapper.status):
    raise HTTPException(detail="Workflow already completed", status_code=409)
if wrapper is None:
    handler_data = await self._load_handler(handler_id)
    if is_status_completed(handler_data.status):
        raise HTTPException(detail="Workflow already completed", status_code=409)
    else:
        # Currently warns and raises 409 "Handler expired"
        logger.warning(f"Handler {handler_id} is running but not in memory.")
        raise HTTPException(detail="Handler expired", status_code=409)
```

**New behavior:**
```python
wrapper = self._handlers.get(handler_id)

# If not in memory, try to reload from persistence
if wrapper is None:
    handler_data = await self._load_handler(handler_id)
    if is_status_completed(handler_data.status):
        raise HTTPException(detail="Workflow already completed", status_code=409)

    # Attempt to reload the workflow
    wrapper = await self._reload_workflow(handler_id)
    if wrapper is None:
        raise HTTPException(detail="Failed to reload workflow", status_code=500)

# Check if completed (could have completed during reload)
if is_status_completed(wrapper.status):
    raise HTTPException(detail="Workflow already completed", status_code=409)

# Continue with event dispatch...
```

---

### Phase 4: Automatic Idle Workflow Release (Optional)

**Configuration options:**
```python
class WorkflowServer:
    def __init__(
        self,
        *,
        # ... existing params ...
        idle_timeout: float | None = None,  # Seconds before auto-release
        max_idle_handlers: int | None = None,  # Max idle handlers before LRU eviction
    ):
```

**Background task for auto-release:**
```python
async def _idle_cleanup_task(self) -> None:
    """Periodically check and release idle workflows."""
    while True:
        await asyncio.sleep(self._idle_check_interval)

        now = datetime.now(timezone.utc)
        to_release = []

        for handler_id, wrapper in self._handlers.items():
            if wrapper.idle_since is None:
                continue

            idle_duration = (now - wrapper.idle_since).total_seconds()
            if idle_duration > self._idle_timeout:
                to_release.append(handler_id)

        for handler_id in to_release:
            try:
                await self.release_idle_workflow(handler_id)
            except Exception as e:
                logger.error(f"Failed to release idle workflow {handler_id}: {e}")
```

---

## Edge Cases Analysis

### 1. Event sent to idle workflow, but event type is not handled

**Scenario:** User sends `POST /events/{handler_id}` with an event type that no step accepts.

**Current behavior:** The event is dispatched via `ctx.send_event(event, step=step)`. If `step` is specified, it validates the step accepts the event type and raises `WorkflowRuntimeError`. If `step` is None, the event is broadcast to all step queues.

**With idle release:**
- Workflow is reloaded from persistence
- Event is dispatched to all step queues
- If no step handles it, the event sits in queues
- **Question: Should the workflow be re-released?**

**Proposed solution:**
- After reloading, track if the workflow becomes idle again quickly (within a small window, e.g., 1 second)
- If it does, consider it still idle and allow it to be released again
- The `idle_since` will be reset naturally when the workflow emits another `InputRequiredEvent`

**Alternative:** Add a "reload grace period" during which the workflow cannot be released, giving it time to process the event.

### 2. Concurrent release and event arrival

**Scenario:** Release operation is in progress when a new event arrives via `POST /events/{handler_id}`.

**Current flow without locking:**
1. Release task starts, calls `wrapper.checkpoint()`
2. Event POST arrives, finds handler in `_handlers`
3. Release task removes handler from `_handlers`
4. Event POST tries to send to context, but handler is being torn down

**Proposed solution:** Add a release lock per handler:
```python
@dataclass
class _WorkflowHandler:
    # ... existing fields ...
    _release_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
```

In `release_idle_workflow`:
```python
async with wrapper._release_lock:
    # Check if still idle (could have received event while waiting for lock)
    if wrapper.idle_since is None:
        return False
    # Proceed with release...
```

In `_post_event` after reload:
```python
async with wrapper._release_lock:
    # Ensure not being released
    ctx.send_event(event, step=step)
```

### 3. Workflow store unavailable during reload

**Scenario:** User sends event, workflow needs reload, but database is down.

**Proposed solution:** Return 503 Service Unavailable with retry-after header:
```python
try:
    wrapper = await self._reload_workflow(handler_id)
except Exception as e:
    logger.error(f"Failed to reload workflow {handler_id}: {e}")
    raise HTTPException(
        detail="Workflow storage unavailable",
        status_code=503,
        headers={"Retry-After": "5"}
    )
```

### 4. Workflow completes immediately after reload

**Scenario:** Workflow is reloaded, but all its work was already done (e.g., timeout expired, or it was waiting for an optional event).

**Current behavior:** Workflow runs, completes, final checkpoint sets status to completed.

**With idle release:** Same behavior, no special handling needed. The workflow will complete and be removed from `_handlers`.

### 5. Multiple events sent rapidly to released workflow

**Scenario:** Client sends multiple events in quick succession to a released workflow.

**Risk:** Multiple concurrent reload attempts.

**Proposed solution:** Use a "loading" state or lock:
```python
_loading_handlers: dict[str, asyncio.Lock] = {}

async def _reload_workflow(self, handler_id: str) -> _WorkflowHandler | None:
    # Get or create load lock
    if handler_id not in self._loading_handlers:
        self._loading_handlers[handler_id] = asyncio.Lock()

    async with self._loading_handlers[handler_id]:
        # Check if already loaded by another request
        if handler_id in self._handlers:
            return self._handlers[handler_id]

        # Proceed with reload...
```

### 6. Streaming endpoint called on released workflow

**Scenario:** Client calls `GET /events/{handler_id}` on a released workflow.

**Current behavior:** Returns 404 "Handler not found" (since it's not in `_handlers`).

**Proposed behavior options:**
1. **Auto-reload and stream:** Reload the workflow and stream events. This may be unexpected behavior.
2. **Return 503 with reload hint:** Return error indicating workflow is released and should be activated first.
3. **Return empty stream:** Return 204 No Content if the workflow has been idle with no new events.

**Recommended:** Option 2 - Return 503 or a new status code indicating the workflow is paused/released and needs to be activated via event POST.

---

## Implementation Order

1. **Phase 1:** Add `idle_since` tracking (minimal, observable change)
2. **Phase 2:** Implement release mechanism (internal, can be manually triggered)
3. **Phase 3:** Auto-reload on event POST (enables the full feature)
4. **Phase 4:** Automatic cleanup (optional, for production optimization)

## Testing Strategy

1. **Unit tests:**
   - Test `idle_since` is set when workflow becomes idle
   - Test `idle_since` is cleared when workflow resumes
   - Test release removes handler from memory but keeps in store
   - Test reload restores handler from store to memory

2. **Integration tests:**
   - Test full cycle: start -> idle -> release -> event POST -> reload -> complete
   - Test concurrent event POST during release
   - Test multiple rapid event POSTs to released workflow
   - Test streaming endpoint behavior on released workflow

3. **Edge case tests:**
   - Test event of unhandled type to reloaded workflow
   - Test reload with unavailable store
   - Test release of non-idle workflow (should fail)
