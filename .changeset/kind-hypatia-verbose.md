---
"llama-index-workflows": patch
---

Fix `Workflow(verbose=True)` being a no-op by adding a `VerboseDecorator` that intercepts `StepStateChanged` events to print step starts and completions
