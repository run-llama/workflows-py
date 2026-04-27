---
"llama-agents-control-plane": patch
"llama-agents-core": patch
---

Add `follow=false` query param on `GET /deployments/{id}/logs` so clients can fetch currently-available logs and exit. The default stays `follow=true`; existing streaming consumers are unchanged.
