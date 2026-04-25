---
"llamactl": patch
---

Revert previous changes, `llamactl serve` now re-exports frontend API keys with public prefixes once again since this is necessary for local dev auth to work.
