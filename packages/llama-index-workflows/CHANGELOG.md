# llama-index-workflows

## 2.14.0

### Minor Changes

- 73c1254: refactor: expand runtime plugin architecture

  - Refactoring to better support alternate distributed backends
  - Some `Context` methods may now raise errors if used in an unexpected context
  - `WorkflowHandler` is no longer a future. Retains compatibility methods for main use cases (exception, cancel, etc)

- 45e7614: Replace InMemoryStateStore types with a corresponding StateStore protocol
- 2900f58: Support state type inheritance in workflows

### Patch Changes

- 45e7614: Refact: make control loop more deterministic

  - Switches out the asyncio delay mechanism for a pull-with-timeout that is more deterministic friendly
  - Adds a priority queue of delayed tasks
  - Switches out the misc firing /spawning of async tasks to a more rigorous pattern where tasks are only created in the main loop, and gathered in one location. This makes the concurrency more straightforward to reason about

- 6fdc45c: Update debugger assets

  - JavaScript: https://cdn.jsdelivr.net/npm/@llamaindex/workflow-debugger@0.2.15/dist/app.js
  - CSS: https://cdn.jsdelivr.net/npm/@llamaindex/workflow-debugger@0.2.15/dist/app.css

## 2.13.1

### Patch Changes

- e958ed2: fix: ResourceConfig was loading config file eagerly
- ebaf212: Add resource validation to workflow.validate()

## 2.13.0

### Minor Changes

- 6dd7fc0: Add resource config node support to workflow representation
- be19869: Add support for injecting resources more flexibly

  - Add support for injecting Resources recursively, so a Resource can depend on another Resource or ResourceConfig
  - Add support for injecting ResourceConfig directly into steps
  - Fix issues with resolving from String quoted types

## 2.12.2

### Patch Changes

- bfbfba4: Return an empty list for empty target events, rather than None
- 85f948e: fix: rebuild_state_from_ticks clears in_progress before replaying

  Fixed ctx.to_dict() failing with "Worker X not found in in_progress" when checkpointing resumed workflows. The function now also rewinds in progress when recreating from ticks, to match the actual behavior when resuming a workflow.

## 2.12.1

### Patch Changes

- 40be1c7: add workflow class name to WorkflowGraph representation

## 2.12.0

### Minor Changes

- e53c654: Add further detail to workflow graph, mainly adding `Resource` nodes to workflow graph and visualizations
- 2ff316d: Updates workflow server with functionality to drop and restore idle workflow handlers that are waiting on external input.
- 0d72b4d: reorganize workflow graph representation types
- f96faa2: Add dedicated StopEvent subclasses for workflow termination (timeout, cancellation, failure)

### Patch Changes

- 3b043b8: Track when workflows are idle (waiting on external input)
- 7a85c96: Add ResourceConfig for resource-level configuration injection

## 2.11.7

### Patch Changes

- 6c35e4d: Update debugger assets

  - JavaScript: https://cdn.jsdelivr.net/npm/@llamaindex/workflow-debugger@0.2.12/dist/app.js
  - CSS: https://cdn.jsdelivr.net/npm/@llamaindex/workflow-debugger@0.2.12/dist/app.css

- f58537a: Update debugger assets

  - JavaScript: https://cdn.jsdelivr.net/npm/@llamaindex/workflow-debugger@0.2.14/dist/app.js
  - CSS: https://cdn.jsdelivr.net/npm/@llamaindex/workflow-debugger@0.2.14/dist/app.css

## 2.11.6

### Patch Changes

- 94fa8ce: Fix infinite retries with no delay
- f8fa366: Update debugger assets

  - JavaScript: https://cdn.jsdelivr.net/npm/@llamaindex/workflow-debugger@0.2.10/dist/app.js
  - CSS: https://cdn.jsdelivr.net/npm/@llamaindex/workflow-debugger@0.2.10/dist/app.css

## 2.11.5

### Patch Changes

- 27a4cf0: Update debugger assets

  - JavaScript: https://cdn.jsdelivr.net/npm/@llamaindex/workflow-debugger@0.2.2/dist/app.js
  - CSS: https://cdn.jsdelivr.net/npm/@llamaindex/workflow-debugger@0.2.2/dist/app.css

## 2.11.4

### Patch Changes

- 95abac0: Update debugger assets

  - JavaScript: https://cdn.jsdelivr.net/npm/@llamaindex/workflow-debugger@0.2.0/dist/app.js
  - CSS: https://cdn.jsdelivr.net/npm/@llamaindex/workflow-debugger@0.2.0/dist/app.css

- 8f344bd: Fix resuming from serialized context for workflows that uses typed events

## 2.11.3

### Patch Changes

- f307253: Update typechecking to support ty
- 91159d7: Moving `_extract_workflow_structure` to its own module in workflow core
- 300fd05: Add stricter ruff formatting checks
- 32ae78a: Switch build backend to uv

## 2.11.2

### Patch Changes

- ee56c97: Fix remove task functionality on \_execute_task, specially when the task has gone missing
