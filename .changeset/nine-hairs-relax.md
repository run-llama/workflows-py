---
"llama-index-utils-workflow": minor
"llama-index-workflows": minor
---

refactor: expand runtime plugin architecture

- Refactoring to better support alternate distributed backends
- Some `Context` methods may now raise errors if used in an unexpected context
- `WorkflowHandler` is no longer a future. Retains compatibility methods for main use cases (exception, cancel, etc)
