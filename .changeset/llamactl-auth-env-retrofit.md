---
"llamactl": minor
---

`auth list`, `auth env list`, and `auth organizations` now support `-o text|json|yaml|wide`. Plain-text tables replace the Rich-styled tables; JSON/YAML output round-trips. `auth list` no longer leaks `api_key` or OIDC tokens — `auth_type` reports `token`/`oidc`/`none`.
