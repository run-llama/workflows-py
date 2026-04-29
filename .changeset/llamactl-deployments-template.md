---
"llamactl": minor
---

Add `llamactl deployments template` (offline scaffolding) and `deployments get -o template`. Both emit apply-shaped YAML annotated with `##` documentation comments. The YAML surface renames `display_name` to `generateName` (the apply parser accepts either key) and renders it as a commented-out slug-seed under the canonical top-level `name`. CLI success messages and the deployment picker print the deployment id instead of the display name.
