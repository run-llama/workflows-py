---
"llama-index-workflows": patch
---

Writing one path with ctx.store.set no longer copies the rest of the state. Only the containers along the written path are rebuilt, so the cost of a write no longer grows with the size of unrelated values, and those values are shared rather than cloned.
