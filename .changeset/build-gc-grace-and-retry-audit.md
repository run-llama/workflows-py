---
"llama-agents-control-plane": patch
"llamactl": patch
---

Add a grace window to build artifact GC (configurable via `BUILD_ARTIFACT_GC_GRACE_SECONDS`, default 75m) and stop auto-retrying `llamactl auth`'s non-idempotent key-creation POST.
