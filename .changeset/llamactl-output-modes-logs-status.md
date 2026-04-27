---
"llama-agents-core": minor
"llama-agents-control-plane": minor
"llamactl": minor
---

llamactl read commands gain `-o text|json|yaml` output modes and a `--project` override. `deployments get` replaces the Textual monitor with a plain table (single-row when given a name), and `deployments list` becomes a hidden alias. `deployments logs` is a standalone subcommand; the server's `/deployments/{id}/logs` endpoint accepts a `follow` query parameter so non-streaming clients fetch recent logs and exit. Structured `get` output is now a CLI-owned shape: top-level editable fields, a nested `status:` block, masked `secrets`, and no leaked deprecated aliases. Tables are plain whitespace (no Rich, no truncation). `auth list -o json` reports `auth_type` as `none`/`token`/`oidc`; `auth env list -o json` no longer includes `min_llamactl_version`.
