---
"llama-agents-core": patch
"llamactl": patch
---

Accept unknown deployment phase values as plain strings instead of failing validation, so older clients keep working when a newer server emits a phase the client doesn't know about.
