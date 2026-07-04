---
"llama-index-workflows": minor
"llama-agents-agentcore": patch
"llama-agents-server": patch
"llama-agents-dbos": patch
"llama-agents-client": patch
---

Add in-process child workflow composition with typed child declarations, namespaced execution, per-child state, catch-error recovery, timeouts, and opt-in child event streaming.

Each child invocation runs as a nested workflow with its own isolated state, streams, and waiters. Overlapping invocations of the same child slot get stable, sequential ids (`child#0`, `child#1`), so targeted sends and stream origins can address a specific invocation (`child#0/answer`). A send to a completed invocation produces an `UnhandledEvent` rather than silently re-entering it, and a static child path (`child/answer`) without a concrete invocation is rejected.

Timeouts now measure known-alive time, not wall-clock, at every level (root and every child). A workflow may spend `timeout` seconds of alive time; time while the process is down between a snapshot and its resume is forgiven, so a resumed run keeps the alive budget it already spent instead of resetting. This changes the previous root-timeout behavior, which reset to a fresh full budget on every resume — a long-idle workflow that resumes will no longer get its whole timeout back.

An uncaught failure or timeout inside a child no longer fails the whole run: it surfaces to the parent as a `StepFailedEvent` named for the child's path, eligible for the parent's `@catch_error` (with normal `max_recoveries`), recursing upward. Only a failure uncaught all the way to the root ends the run.

Annotated Workflow attributes are only auto-attached when they use the typed child workflow contract, so existing manual composition with bare StartEvent/StopEvent workflows remains compatible.

Child workflows run on the durable server, DBOS, and AgentCore runtimes, not just in-process. Each invocation gets its own persistent state record keyed by `(run_id, namespace)`, so per-child state survives idle release and restart and resumes with a suspended child waiter intact. On DBOS a child runs inside the parent's durable workflow rather than as a separate one. Child-origin events carry their invocation namespace over the SSE stream and are hidden unless the reader opts in with `stream_events(include_children=True)` (server query param `include_children=true`).
