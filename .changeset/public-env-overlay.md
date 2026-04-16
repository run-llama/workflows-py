---
"llama-agents-appserver": minor
"llamactl": patch
---

`PUBLIC_*` env var overlay for UI builds: `PUBLIC_X` overrides `X` in the build env so backend and frontend can use different URLs for the same service. Removes dead `VITE_`/`NEXT_PUBLIC_` injection from `llamactl serve`. Helm network policy gains `extraEgressRules`, DNS selector overrides, and `blockPrivateRanges` toggle.
