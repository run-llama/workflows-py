# DBOS Adapter Architecture

## Model Overview

DBOS is a **local runtime with database coordination**. Each DBOS process runs workflows and steps co-located in the same process. Coordination between replicas happens through a shared Postgres database.

There are no distributed step workers — a workflow and all its steps execute in the same process that started them.

## Executor ID and Workflow Ownership

Each DBOS replica is configured with a unique `executor_id` (e.g. `"replica-8001"`). This ID is recorded in the database for every workflow the replica starts, creating a natural ownership model:

- A replica **owns** the workflows it started. On startup, DBOS automatically recovers and relaunches any incomplete workflows belonging to its `executor_id`.
- This makes each replica a **stateful shard** — it's responsible for a specific subset of workflows, determined by which ones were routed to it.
- Replicas share the same Postgres database and can communicate through it, but each replica only runs its own workflows.

The `executor_id` model means horizontal scaling works by adding replicas that each own a slice of the workload, not by distributing individual workflow steps across nodes.

## Process Layout

```
┌──────── Replica A (executor_id: replica-8001) ────────┐
│  WorkflowServer                                       │
│  ├─ DBOSRuntime                                       │
│  │  ├─ InternalAdapter (per run) ← runs the workflow  │
│  │  └─ Workflow steps (co-located)                    │
│  └─ ExternalAdapter (per run) ← receives HTTP calls   │
└───────────────────────┬───────────────────────────────┘
                        │
                   Shared Postgres
                   (DBOS tables + pg_notify)
                        │
┌──────── Replica B (executor_id: replica-8002) ────────┐
│  WorkflowServer                                       │
│  ├─ DBOSRuntime                                       │
│  │  ├─ InternalAdapter (per run)                      │
│  │  └─ Workflow steps (co-located)                    │
│  └─ ExternalAdapter (per run)                         │
└───────────────────────────────────────────────────────┘
```

## The Adapter Boundary

The core runtime exposes two adapter interfaces per workflow run:

- **InternalRunAdapter** — used by the control loop running the workflow. Always in the same process as the workflow.
- **ExternalRunAdapter** — used by callers (HTTP handlers, other services). May be in a different process.

For DBOS, this distinction maps to a **process boundary**. The internal adapter uses `DBOS.send()` / `DBOS.recv_async()` locally. The external adapter uses `DBOS.send_async()` which writes to Postgres, making it reachable from any replica.

## Event Delivery (Cross-Process)

When Replica B sends an event to a workflow owned by Replica A:

1. Replica B's external adapter writes the event to Postgres via `DBOS.send_async()`
2. Replica A's internal adapter picks it up via `DBOS.recv_async()` (polls Postgres)
3. The event is delivered to the workflow's control loop in Replica A

## Event Streaming (Cross-Process)

Workflow output events flow through `WorkflowStore` backed by Postgres:

1. A workflow step publishes an event via `write_to_event_stream()`
2. The store writes to Postgres and sends `pg_notify`
3. Any replica calling `subscribe_events(run_id)` receives the event

## Idle Release (Continue-as-New)

`DBOSIdleReleaseDecorator` wraps the runtime to release idle workflows from memory using a "continue-as-new" approach:

- **Release**: When a workflow goes idle, a timer starts. After `idle_timeout` seconds, the decorator sends `TickIdleRelease` through the external adapter via `DBOS.send_async()`. The internal adapter picks it up via `DBOS.recv_async()`, and the control loop processes it, cleanly completing the workflow with an `IdleReleasedEvent`. DBOS marks the workflow as SUCCESS. We then purge the DBOS state and journal entries so the run_id can be reused on resume (see the docstring in `idle_release.py` for details).
- **Resume**: When an event arrives for an idle-released handler (`idle_since` is set), the decorator rebuilds `BrokerState` from the tick log, purges stale DBOS/journal state, and starts a fresh DBOS workflow reusing the same `run_id` with the rebuilt state. The handler in the store keeps the same `run_id` with `idle_since=None`.

Tick persistence is provided by `TickPersistenceDecorator` in the decorator chain, which stores ticks to the workflow store so they can be replayed on resume.

Both operations go through the database, so any replica can resume an idle-released workflow — the new DBOS workflow starts on whichever replica handles the incoming event.

## Guidelines for DBOS Code

**Process boundary awareness**: External adapter methods may execute in a different process from the workflow. They must communicate exclusively through the database — no local state, no asyncio task references. Internal adapter methods are co-located with the workflow and can use process-local state when needed.

**Don't cancel workflows on shutdown**: DBOS automatically recovers incomplete workflows belonging to the replica's `executor_id` on startup. Cancelling them during shutdown would prevent recovery.

**Use asyncio for process-local coordination**: DBOS durable messages persist in the DB and replay on recovery. Don't use them for ephemeral control flow — use normal asyncio primitives instead.

**Be aware of the replica model**: Code that assumes single-process (e.g. in-memory tracking of all active runs, direct asyncio Future manipulation across adapter boundaries) will break when replicas are involved. Always consider whether the code path might cross a process boundary.
