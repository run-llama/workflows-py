---
"llama-index-workflows": minor
---

Add composable retry primitives (`ComposableRetryPolicy`, `retry_if_exception_type`, `wait_exponential`, `stop_after_attempt`, etc.). The existing `RetryPolicy` name remains the structural protocol, preserving backward compatibility for custom retry policies.
