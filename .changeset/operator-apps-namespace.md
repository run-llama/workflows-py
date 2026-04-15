---
"llama-agents": minor
---

Operator supports a dedicated namespace for `LlamaDeployment` CRs and their app resources via `operator.apps.namespace`. When set, the operator + control plane stay in the release namespace while CRs and child Deployments/Services/Secrets/Jobs live in the apps namespace. Empty preserves the existing single-namespace behaviour.
