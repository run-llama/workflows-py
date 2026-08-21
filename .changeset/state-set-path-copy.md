---
"llama-index-workflows": patch
---

ctx.store.set now rebuilds only the written path instead of copying the whole state, so the cost of a write no longer grows with the size of unrelated values.
