---
"llama-agents-dbos": minor
"llama-agents-server": minor
---

DBOSRuntime, PostgresWorkflowStore, and the executor lease manager now share a single asyncpg pool per runtime instead of each opening their own. The pool size is configurable via `DBOSRuntimeConfig.pool_size` / `pool_min_size` and defaults to DBOS's configured `sys_db` pool size. The `PostgresWorkflowStore` LISTEN connection also reconnects automatically with bounded backoff when the underlying connection drops.
