---
"llama-agents-server": patch
"llama-index-workflows": patch
---

Refact: make control loop more deterministic

-   Switches out the asyncio delay mechanism for a pull-with-timeout that is more deterministic friendly
-   Adds a priority queue of delayed tasks
-   Switches out the misc firing /spawning of async tasks to a more rigorous pattern where tasks are only created in the main loop, and gathered in one location. This makes the concurrency more straightforward to reason about
