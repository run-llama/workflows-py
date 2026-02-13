# llama-agents-server

## 0.2.0-rc.2

### Minor Changes

- 6ccdebd: Refactor server internals from monolithic handler to composable runtime decorators (ServerRuntimeDecorator, PersistenceDecorator, IdleReleaseDecorator) enabling pluggable server runtimes

## 0.2.0-rc.1

### Minor Changes

- 528d562: Add tick storage, event storage with SSE subscription, per-run state stores, and centralized handler status transitions to AbstractWorkflowStore and SQLite/memory implementations

### Patch Changes

- Updated dependencies [528d562]
- Updated dependencies [e981f73]
- Updated dependencies [b515a46]
- Updated dependencies [7433d4c]
  - llama-agents-client@0.2.0-rc.0
  - llama-index-workflows@2.15.0-rc.0

## 0.2.0-rc.0

### Minor Changes

- 06cca76: Test pre-release functioning

## 0.1.2

### Patch Changes

- ef7f808: Fix OpenAPI schema version to use current server package, not workflows core
- Updated dependencies [6ece797]
  - llama-index-workflows@2.14.1
  - llama-agents-client@0.1.2

## 0.1.1

### Patch Changes

- db90f89: Separate server/client to their own packages under a llama_agents namespace
- 45e7614: Refact: make control loop more deterministic

  - Switches out the asyncio delay mechanism for a pull-with-timeout that is more deterministic friendly
  - Adds a priority queue of delayed tasks
  - Switches out the misc firing /spawning of async tasks to a more rigorous pattern where tasks are only created in the main loop, and gathered in one location. This makes the concurrency more straightforward to reason about

- Updated dependencies [db90f89]
- Updated dependencies [73c1254]
- Updated dependencies [45e7614]
- Updated dependencies [45e7614]
- Updated dependencies [2900f58]
- Updated dependencies [6fdc45c]
  - llama-agents-client@0.1.1
  - llama-index-workflows@2.14.0
