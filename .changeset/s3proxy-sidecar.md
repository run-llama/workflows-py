---
"llama-agents": minor
---

Add optional `s3proxy` sidecar for the control plane. Set `s3proxy.enabled=true` and fill in `s3proxy.config` (JCLOUDS_* env vars) to run llama-agents on Azure Blob, GCS, or any other jclouds-supported backend without standing up a separate proxy. See `charts/llama-agents/docs/s3-proxy-setup.md`.
