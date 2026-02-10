# Core Architecture: Workflow, Context, and Runtime

## Overview

```mermaid
graph LR
    User -->|".run()"| Workflow
    Workflow -->|creates| Context
    Context -->|"_workflow_run()"| Runtime
    Runtime -->|launches| ControlLoop
    Runtime -->|returns| Handler[WorkflowHandler]
    ControlLoop -->|uses| InternalAdapter[InternalRunAdapter]
    Handler -->|uses| ExternalAdapter[ExternalRunAdapter]
```

A **Workflow** is a container for step functions. Calling `run()` creates a **Context**, which asks the **Runtime** to launch execution. The runtime returns two adapters — one for the **control loop** (internal) and one for the caller via **WorkflowHandler** (external).

## Workflow

Container for step definitions. A `WorkflowMeta` metaclass collects `@step`-decorated methods into `_step_functions`.

`run()` selects a runtime (explicit param > ContextVar > basic_runtime), validates steps, then delegates to `Context._workflow_run()`.

[`workflow.py`](../packages/llama-index-workflows/src/workflows/workflow.py)

## Context Faces

Context presents different interfaces depending on execution phase. Internally it holds a `_face` field that transitions through three types:

```mermaid
graph LR
    Pre[PreContext] -->|"_workflow_run()"| External[ExternalContext]
    External -.->|"per step worker"| Internal[InternalContext]
```

| Face | When | Used By | Key Operations |
|------|------|---------|----------------|
| PreContext | Before `run()` | Setup code | Configuration, serialization, state store init |
| ExternalContext | After `run()` | Handler / caller | `send_event()`, `stream_events()` |
| InternalContext | During step execution | Step functions | `collect_events()`, `wait_for_event()`, `write_event_to_stream()` |

Each face wraps an adapter from the runtime. Public methods on Context check the current face and raise `ContextStateError` if called in the wrong phase.

- [`context.py`](../packages/llama-index-workflows/src/workflows/context/context.py) — Context, `_face` field
- [`pre_context.py`](../packages/llama-index-workflows/src/workflows/context/pre_context.py) — PreContext
- [`external_context.py`](../packages/llama-index-workflows/src/workflows/context/external_context.py) — ExternalContext
- [`internal_context.py`](../packages/llama-index-workflows/src/workflows/context/internal_context.py) — InternalContext

## Runtime and Adapters

The `Runtime` ABC uses a dual-adapter pattern. Each workflow run produces two adapters sharing a `run_id`:

```mermaid
graph TB
    Runtime -->|"run_workflow()"| ExternalRunAdapter
    Runtime -->|"get_internal_adapter()"| InternalRunAdapter
    ExternalRunAdapter --- run_id
    InternalRunAdapter --- run_id
```

| Adapter | Used By | Key Methods |
|---------|---------|-------------|
| InternalRunAdapter | Control loop | `wait_receive()`, `on_tick()`, `wait_for_next_task()`, `get_now()` |
| ExternalRunAdapter | WorkflowHandler | `send_event()`, `stream_published_events()`, `get_result()`, `cancel()` |

Runtimes are composable via decorators — see [server-architecture.md](./server-architecture.md#runtime-decorator-chain) for the decorator chain pattern.

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
        IA -->|TickAddEvent| CL[Control Loop]
    end
```

```mermaid
graph LR
    subgraph "Events Out (internal to external)"
        CL2[Control Loop] -->|CommandPublishEvent| IA2[InternalRunAdapter]
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
