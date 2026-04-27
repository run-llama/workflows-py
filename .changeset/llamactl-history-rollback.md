---
"llamactl": minor
---

`deployments history` now supports `-o text|json|yaml|wide` and `--project <id>`. Text output uses 7-char short SHAs and Z-suffixed UTC timestamps; JSON keeps full SHAs. `deployments rollback --git-sha` now offers shell completion from the deployment's history.
