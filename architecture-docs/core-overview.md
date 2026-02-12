# Core Architecture: Workflow, Context, and Runtime

## Overview

```mermaid
graph LR
    subgraph Client ["Client (public API)"]
        WR["Workflow.run()"]
        WH[WorkflowHandler]
        EC["Context\n(ExternalContext)"]
    end

    subgraph Runtime
        EA[ExternalRunAdapter]
        CL[Control Loop]
        IA[InternalRunAdapter]
    end

    subgraph Worker ["Worker (per step)"]
        IC["Context\n(InternalContext)"]
        Steps[Step Functions]
    end

    WR -->|launches| CL
    WH --- EA
    EC --- EA
    EA --- CL
    CL --- IA
    IA --- IC
    IA --- Steps
```

The system has three zones. **Client** code calls `Workflow.run()` and interacts through `WorkflowHandler` and `Context`. The **Runtime** sits in the middle — the control loop drives execution, with `ExternalRunAdapter` and `InternalRunAdapter` as its boundaries. **Workers** are step functions that see `Context` with a different face (InternalContext).

The adapters are the key abstraction boundary. Everything on the client side goes through `ExternalRunAdapter`; everything on the worker side goes through `InternalRunAdapter`. The runtime is swappable by replacing or decorating these adapters.

## Workflow

Container for step definitions. `run()` selects a runtime, validates steps, and delegates to the runtime for execution.

[`workflow.py`](../packages/llama-index-workflows/src/workflows/workflow.py)

## Context Faces

Context presents different interfaces depending on execution phase. Internally it holds a `_face` field that transitions through three types:

```mermaid
graph LR
    Pre[PreContext] -->|"_workflow_run()"| External[ExternalContext]
    External -.->|"per step worker"| Internal[InternalContext]
```

| Face | When | Used By |
|------|------|---------|
| PreContext | Before `run()` | Setup code — configuration, serialization, state store init |
| ExternalContext | After `run()` | Handler / caller — sending events, streaming |
| InternalContext | During step execution | Step functions — collecting events, publishing to stream |

Each face wraps an adapter from the runtime. Public methods on Context check the current face and raise `ContextStateError` if called in the wrong phase.

[`context/`](../packages/llama-index-workflows/src/workflows/context/) — Context implementation and all face types

## Runtime and Adapters

The `Runtime` ABC uses a dual-adapter pattern. Each workflow run produces two adapters sharing a `run_id`:

```mermaid
graph TB
    Runtime -->|"run_workflow()"| ExternalRunAdapter
    Runtime -->|"get_internal_adapter()"| InternalRunAdapter
    ExternalRunAdapter --- run_id
    InternalRunAdapter --- run_id
```

The **InternalRunAdapter** is used by the control loop — it handles receiving ticks, publishing events, timing, and task coordination. The **ExternalRunAdapter** is used by the WorkflowHandler — it handles sending events in, streaming published events out, getting results, and cancellation.

Multiple base runtimes exist. `BasicRuntime` is the default in-memory asyncio runtime in the core package. `DBOSRuntime` provides durable distributed execution backed by a database. More runtimes can be added by implementing the `Runtime` ABC. Runtimes are also composable via decorators — see [server-architecture.md](./server-architecture.md#runtime-decorator-chain) for that pattern.

[`plugin.py`](../packages/llama-index-workflows/src/workflows/runtime/types/plugin.py) — Runtime ABC, InternalRunAdapter, ExternalRunAdapter

## Control Loop

The control loop is the core execution engine. It follows a reducer pattern where pure state transitions produce side effects as commands. The control loop is runtime-agnostic — it interacts with the outside world exclusively through `InternalRunAdapter`.

See [control-loop.md](./control-loop.md) for the full architecture.

## Event Flow

```mermaid
graph LR
    subgraph "Events In (external to internal)"
        H[WorkflowHandler] -->|send_event| EA[ExternalRunAdapter]
        EA -->|receive queue| IA[InternalRunAdapter]
        IA -->|tick| CL[Control Loop]
    end
```

```mermaid
graph LR
    subgraph "Events Out (internal to external)"
        CL2[Control Loop] -->|command| IA2[InternalRunAdapter]
        IA2 -->|publish queue| EA2[ExternalRunAdapter]
        EA2 -->|stream_published_events| H2[WorkflowHandler]
    end
```

## WorkflowHandler

Returned by `Workflow.run()`. The user-facing handle for a running workflow.

- `await handler` — blocks until StopEvent, returns the result
- `handler.stream_events()` — async iterator of published events (single consumption)
- `handler.send_event()` — send events into the running workflow
- `handler.cancel_run()` — graceful cancellation

[`handler.py`](../packages/llama-index-workflows/src/workflows/handler.py)
