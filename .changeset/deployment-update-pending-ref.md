---
"llama-agents-control-plane": patch
---

Updating a push-mode deployment to a branch that hasn't been pushed yet now succeeds and marks the deployment as pending; the next push to that ref bootstraps the SHA instead of failing with 400.
