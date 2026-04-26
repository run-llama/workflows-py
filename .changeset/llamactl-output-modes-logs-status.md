---
"llama-agents-core": minor
"llama-agents-control-plane": minor
"llamactl": minor
---

Add `-o text|json|yaml` output modes and `--project` override to llamactl read commands, split the Textual deployment monitor into standalone `deployments logs` and `deployments status` subcommands, and add a `follow` query parameter to `/deployments/{id}/logs` so non-streaming clients can fetch recent logs and exit. `deployments get` now prints info instead of launching the TUI; the former `deployments list` becomes a hidden alias.
