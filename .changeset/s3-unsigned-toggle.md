---
"llama-agents-control-plane": patch
---

Add `controlPlane.objectStorage.s3.unsigned` (Helm) / `S3_UNSIGNED` (env) toggle to send S3 requests unsigned, enabling authless S3-compatible backends (s3proxy, LocalStack, public-read buckets) without placeholder credentials. When enabled, applies to all S3 uses — builds, backups, and code-repo storage.
