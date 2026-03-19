# Workflows API reference


For conceptual guides, patterns, and end-to-end examples, see the main [Agent Workflows documentation](/python/llamaagents/workflows/). This API reference focuses on signatures, parameters, and behavior of individual classes and functions.

## Installation

Install the core Workflows SDK from PyPI:

```bash
pip install llama-index-workflows
```

## What this reference covers

- **Workflow**: The event-driven orchestrator (`Workflow`) that defines and runs your application flows using typed steps. See [`Workflow`](./workflow/).
- **Context**: Execution context and state management across steps and runs.
- **Events**: Typed events that steps receive and emit (including `StartEvent`, `StopEvent`, and custom events).
- **Decorators**: The `@step` decorator and related utilities for defining workflow steps.
- **Handler**: `WorkflowHandler`, used to await results and stream intermediate events.
- **Retry policy**: `RetryPolicy` configuration for per-step retry behavior.
- **Errors**: Exception types raised by the runtime (validation, configuration, and runtime errors).
- **Resource**: Resource and dependency injection primitives used by steps.

Use this reference together with the live examples on the main docs site, such as the [`Workflow` page on developers.llamaindex.ai](https://developers.llamaindex.ai/python/workflows-api-reference/workflow/), when you need both conceptual context and exact API details.
