---
"llamactl": patch
---

`llamactl serve` now injects `PUBLIC_`, `VITE_`, and `NEXT_PUBLIC_` prefixed copies of `LLAMA_CLOUD_API_KEY` and `LLAMA_CLOUD_BASE_URL` for local dev. `PUBLIC_` is the canonical prefix templates should read; production deployments must opt in explicitly rather than always exposing the token to client code.
