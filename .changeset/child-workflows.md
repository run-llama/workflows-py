---
"llama-index-workflows": minor
"llama-agents-agentcore": patch
"llama-agents-server": patch
"llama-agents-dbos": patch
---

Add in-process child workflow composition with typed child declarations, namespaced execution, per-child state, catch-error recovery, timeouts, and opt-in child event streaming.

Each child invocation runs as a nested workflow with its own isolated state, streams, and waiters. Overlapping invocations of the same child slot get stable, sequential ids (`child#0`, `child#1`), so targeted sends and stream origins can address a specific invocation (`child#0/answer`). A send to a completed invocation produces an `UnhandledEvent` rather than silently re-entering it, and a static child path (`child/answer`) without a concrete invocation is rejected.

Annotated Workflow attributes are only auto-attached when they use the typed child workflow contract, so existing manual composition with bare StartEvent/StopEvent workflows remains compatible.

Update server, DBOS, and AgentCore runtime adapter compatibility for namespaced step IDs and state-store access.
