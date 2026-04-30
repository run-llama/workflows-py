---
"llamactl": patch
---

Fix `deployments update` crashing with `Event loop is closed` after a transient failure on the internal git push. The command now runs `get_deployment` and `update_deployment` in a single event loop instead of reusing the same `ProjectClient` across two `asyncio.run` calls.
