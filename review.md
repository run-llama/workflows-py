• Findings

  - High: Idle release can leave the workflow runtime alive, so “released”
    handlers still consume memory and may keep background tasks running.
    _release_handler only cancels the server’s stream task and removes the
    wrapper from _handlers, but it never stops the underlying
    WorkflowHandler/broker (packages/llama-index-workflows/src/workflows/
    server/server.py:1417). This undermines the memory‑release goal and
    creates a leak surface if the runtime keeps queues/timers alive.
  - High: Idle handlers can hang around indefinitely if a stream consumer
    is active when the timer fires. _start_idle_release_timer skips
    release when consumer_mutex is locked, but it does not reschedule
    another timer when the lock is later released (packages/llama-index-
    workflows/src/workflows/server/server.py:1655). Result: if a client
    briefly held the stream and disconnected after the timeout, the
    handler is never released.
  - High: Race window where an incoming event wakes a workflow but the
    idle timer still fires before StepStateChanged(RUNNING) is observed.
    Because idle_since is only cleared inside the stream consumer, the
    timer can release the handler right after an event is posted but
    before the “running” event reaches _stream_events (packages/llama-
    index-workflows/src/workflows/server/server.py:1692). This can drop
    the in‑memory handler while the workflow is active, leading to missing
    persistence updates and stuck handlers.
  - Medium: Concurrent reloads can create multiple in‑memory handlers for
    the same handler_id. _try_reload_handler has no per‑handler lock or
    “claim” mechanism; two parallel requests can both reload and start a
    workflow (packages/llama-index-workflows/src/workflows/server/
    server.py:1438). In a future replica setup, this becomes a real
    split‑brain risk because each replica can reload the same handler and
    process events independently.
  - Medium: Backward‑compatibility/behavior change: idle release is
    enabled by default with a 10‑second timeout (packages/llama-index-
    workflows/src/workflows/server/server.py:70). This changes semantics
    for existing deployments (e.g., workflows will now be evicted even
    without explicit opt‑in). Also, HandlerQuery now requires store
    support for is_idle; custom store implementations may need updates to
    avoid mismatched behavior (packages/llama-index-workflows/src/
    workflows/server/abstract_workflow_store.py:14).

  Questions / Assumptions

  - Do we want idle release to be opt‑in (default None) until replica
    support and lease/claim semantics exist? If not, what guarantees
    should users expect about in‑memory presence and event ordering during
    eviction?

  Change Summary

  - Introduces idle tracking, persistence of idle_since, and an
    idle‑release timer that can evict handlers and reload them on demand.
    Adds is_idle query filtering and sqlite migration for idle_since.

  Suggested Next Steps

  1. Decide how to fully release runtimes: cancel broker tasks or
     introduce a “hibernate” path that persists state and shuts down the
     runtime cleanly.
  2. Add a reschedule path when release is skipped due to consumer_mutex
     and add a guard (or lock) to prevent release during an in‑flight
     event wake‑up.
  3. Add per‑handler locking/claiming in _try_reload_handler to prevent
     duplicate reloads (especially important for future replica support).
