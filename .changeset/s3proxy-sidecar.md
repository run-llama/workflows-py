---
"llama-agents": minor
---

Add optional `s3proxy` sidecar for the control plane and align both the sidecar and the control plane S3 credentials on a single inline-or-BYO shape. Set `s3proxy.enabled=true` and fill in `s3proxy.config` (JCLOUDS_* env vars) to run llama-agents on Azure Blob, GCS, or any other jclouds-supported backend, or point `s3proxy.secret` at an existing Secret to skip the chart-rendered one. Control plane S3 creds now also accept inline `controlPlane.objectStorage.s3.accessKey`/`secretKey` or a BYO `controlPlane.objectStorage.s3.secret`; the old outer-level `controlPlane.objectStorage.secretRef` keeps working as a silent alias. See `charts/llama-agents/docs/s3-proxy-setup.md`.
