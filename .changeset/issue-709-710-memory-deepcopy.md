---
"llama-index-workflows": patch
---

Stop crashing on `ctx.store` edits when state holds non-deepcopyable live objects (e.g. a `Memory` or an LLM client). Such values are now preserved by reference during edit isolation instead of raising `TypeError: cannot pickle ...`.
