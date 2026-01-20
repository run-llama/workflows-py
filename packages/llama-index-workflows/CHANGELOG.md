# llama-index-workflows

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
