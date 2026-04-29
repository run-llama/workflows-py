---
"llamactl": minor
"llama-agents-core": minor
"llama-agents-control-plane": minor
---

`deployments apply -f` upserts a deployment from YAML; `delete -f` deletes by name read from a file. Adds `PUT /api/v1beta1/deployments/{id}` (create-or-update) and a `DeploymentApply` schema.
