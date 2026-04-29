---
"llama-agents-dbos": minor
"llama-agents-server": minor
---

Share a single asyncpg pool across DBOSRuntime, PostgresWorkflowStore, and ExecutorLeaseManager instead of each opening their own. Pool size is configurable via `DBOSRuntimeConfig.pool_size`. Also adds LISTEN reconnect with backoff to PostgresWorkflowStore.
