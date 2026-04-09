---
"llama-agents-control-plane": patch
"llamactl": patch
---

Add a grace window to build artifact GC (configurable via `BUILD_ARTIFACT_GC_GRACE_SECONDS`, default 75m) and parallelize its delete loop with bounded concurrency. `llamactl auth`'s non-idempotent key-creation POST now only retries on connect-phase errors (`ConnectError`, `ConnectTimeout`, `PoolTimeout`) so initial-connectivity blips are absorbed without risking duplicate keys from a read-timeout retry.
