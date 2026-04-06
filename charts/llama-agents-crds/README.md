# llama-agents-crds

Helm chart that manages the CRD lifecycle for llama-agents (`LlamaDeployment`, `LlamaDeploymentTemplate`).

## Why a separate CRD chart?

The main `llama-agents` chart places CRDs in the Helm `crds/` directory, which means Helm installs them on `helm install` but **never touches them on `helm upgrade` or `helm uninstall`**. This is the safest default — accidental CRD deletion cascades to all custom resources.

However, when CRD schemas change (e.g., the operator adds new fields), you need a way to upgrade them. This chart puts CRDs in `templates/` so they participate in `helm upgrade`, and uses the `helm.sh/resource-policy: keep` annotation so `helm uninstall` leaves them in place.

## Usage

Install before upgrading the main chart if CRD schema has changed:

```bash
helm upgrade --install llama-agents-crds charts/llama-agents-crds
```

## Uninstall behavior

`helm uninstall llama-agents-crds` will **NOT** delete the CRDs thanks to `helm.sh/resource-policy: keep`.

To truly remove CRDs (WARNING: this deletes all CRs of these types):

```bash
kubectl delete crd llamadeployments.deploy.llamaindex.ai llamadeploymenttemplates.deploy.llamaindex.ai
```
