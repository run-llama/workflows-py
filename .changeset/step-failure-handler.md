---
"llama-index-workflows": minor
---

Add `@catch_error` handler (supports `for_steps=[...]` and `max_recoveries`) and `Context.retry_info()` for handling exhausted step retries inline. `retry_info().last_exception` and `StepFailedEvent.exception` are live Python exceptions.
