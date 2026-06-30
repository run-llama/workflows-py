---
"llama-index-workflows": patch
---

Stop crashing on ctx.store edits when state holds non-deepcopyable objects like memory or LLM clients.
