---
"llamactl": minor
---

`deployments get` now does double duty: with a name it shows that deployment (no auto-launching the TUI); without one it prints a kubectl-style table of all deployments. Adds `-o text|json|yaml|wide`, `--project <id>` to override the active profile, friendlier 404 messages, and a new `deployments logs` command (`--follow`/`--json`/`--tail`/`--since-seconds`/`--include-init-containers`). The old `deployments list` is kept as a hidden alias.
