---
"llamactl": patch
---

`llamactl serve` now exports `PUBLIC_LLAMA_CLOUD_API_KEY` and `PUBLIC_LLAMA_CLOUD_BASE_URL` (and `VITE_` / `NEXT_PUBLIC_` variants) so frontend templates can read cloud credentials in local dev.
