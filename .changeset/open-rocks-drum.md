---
"llamactl": patch
---

Restore VITE_/NEXT_PUBLIC_ prefixed LLAMA_CLOUD_* env vars in `llamactl serve` so Vite and Next.js dev servers can pick up local auth credentials.
