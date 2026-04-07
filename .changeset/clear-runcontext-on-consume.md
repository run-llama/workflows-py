---
"llama-index-workflows": patch
---

fix memory leak where asyncio timers could capture a Workflow reference via RunContext
