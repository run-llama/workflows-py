---
"llama-index-workflows": patch
---

fix: drop RunContext strong refs when the control loop consumes them, so asyncio TimerHandle Context snapshots (e.g. aiohttp's periodic `_cleanup_closed`) cannot pin the workflow graph
