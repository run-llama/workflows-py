# llama-agents-server

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
