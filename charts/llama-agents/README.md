# llama-agents

A Helm chart for deploying Llama Agents (control plane + operator)

## Architecture

This chart deploys two components:

- **Control plane** — API server for managing deployments, builds, and backups
- **Operator** — Kubernetes controller that reconciles `LlamaDeployment` custom resources into running pods

CRDs (`LlamaDeployment`, `LlamaDeploymentTemplate`) are included in the chart's `crds/` directory and installed automatically on first `helm install`. They are **not** modified on upgrade or removed on uninstall (standard Helm CRD behavior).

For managed CRD upgrades, use the companion [`llama-agents-crds`](../llama-agents-crds/) chart. Each release of `llama-agents` pins the compatible CRD chart version in `crds.version` (see the values table below) — use that version when installing or upgrading the CRD chart.

## Prerequisites

- Kubernetes 1.26+
- Helm 3.x
- S3-compatible object storage (for build artifacts and backups)

## Installation

### Fresh install

```bash
helm install llama-agents oci://docker.io/llamaindex/llama-agents \
  --set controlPlane.objectStorage.s3.bucket=my-bucket \
  --set controlPlane.objectStorage.s3.region=us-east-1
```

CRDs are installed automatically from the `crds/` directory.

### With separate CRD management

If you prefer explicit CRD lifecycle management (recommended for production):

```bash
# Install CRD chart first — pin to the compatible version from `crds.version` below
helm install llama-agents-crds oci://docker.io/llamaindex/llama-agents-crds --version <crds.version>

# Install main chart, skipping bundled CRDs
helm install llama-agents oci://docker.io/llamaindex/llama-agents --skip-crds \
  --set controlPlane.objectStorage.s3.bucket=my-bucket
```

## Upgrading

```bash
# If CRD schema has changed, upgrade CRDs first — pin to `crds.version` from the values table
helm upgrade --install llama-agents-crds oci://docker.io/llamaindex/llama-agents-crds --version <crds.version>

# Then upgrade the main chart
helm upgrade llama-agents oci://docker.io/llamaindex/llama-agents
```

## Apps namespace

Set `apps.namespace` to isolate `LlamaDeployment` CRs and their child resources
in a separate namespace. The operator + control plane stay in the release
namespace and target the apps namespace for all app resources.

```bash
kubectl create namespace llama-agents-apps
helm install llama-agents oci://docker.io/llamaindex/llama-agents \
  --namespace llama-agents \
  --set apps.namespace=llama-agents-apps \
  --set controlPlane.objectStorage.s3.bucket=my-bucket
```

`imagePullSecrets` are not mirrored — provision them in the apps namespace
yourself, or use node-level pull credentials. Switching modes on an existing
install requires draining and recreating `LlamaDeployment` CRs.

## Values

### Metrics

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| metrics.enabled | bool | `false` | Enable Prometheus ServiceMonitors |
| metrics.scrapeInterval | string | `"30s"` | Scrape interval for ServiceMonitors |
| metrics.scrapeTimeout | string | `"10s"` | Scrape timeout for ServiceMonitors |
| metrics.additionalMonitorLabels | object | `{}` | Extra labels added to ServiceMonitors for Prometheus discovery (e.g., `release: prometheus`) |

### Images

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| images.controlPlane.repository | string | `"llamaindex/llama-agents-control-plane"` | Control plane image repository |
| images.controlPlane.tag | string | `"0.11.1"` | Control plane image tag |
| images.controlPlane.pullPolicy | string | `"IfNotPresent"` | Control plane image pull policy |
| images.operator.repository | string | `"llamaindex/llama-agents-operator"` | Operator image repository |
| images.operator.tag | string | `"0.11.1"` | Operator image tag |
| images.operator.pullPolicy | string | `"IfNotPresent"` | Operator image pull policy |
| images.appserver.repository | string | `"llamaindex/llama-agents-appserver"` | Appserver image repository (used by operator for managed pods) |
| images.appserver.tag | string | `"0.11.1"` | Appserver image tag |
| images.appserver.pullPolicy | string | `"IfNotPresent"` | Appserver image pull policy |
| images.nginx.repository | string | `"nginxinc/nginx-unprivileged"` | Nginx sidecar image repository |
| images.nginx.tag | string | `"1.27-alpine"` | Nginx sidecar image tag |
| images.nginx.pullPolicy | string | `"IfNotPresent"` | Nginx sidecar image pull policy |

### Control Plane

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| controlPlane.replicas | int | `1` | Number of control plane replicas |
| controlPlane.container.port | int | `8000` | Control plane API port |
| controlPlane.container.env | list | `[]` | Extra environment variables for the control plane container |
| controlPlane.container.envFrom | list | `[]` | Extra envFrom sources (secretRef, configMapRef) for the control plane container |
| controlPlane.container.resources | object | `{requests: {cpu: 100m, memory: 256Mi, ephemeral-storage: 500Mi}}` | Resource requests/limits for the control plane container |
| controlPlane.container.startupProbe | object | `{}` | Startup probe configuration |
| controlPlane.container.livenessProbe | object | `{}` | Liveness probe configuration |
| controlPlane.deployment.annotations | object | `{}` | Annotations for the control plane Deployment |
| controlPlane.deployment.podAnnotations | object | `{}` | Annotations for the control plane pod template |
| controlPlane.service.type | string | `"ClusterIP"` | Control plane Service type |
| controlPlane.service.port | int | `80` | Control plane Service port |
| controlPlane.service.annotations | object | `{}` | Annotations for the control plane Service |
| controlPlane.service.metricsPath | string | `"/metrics"` | Metrics path for the control plane Service |
| controlPlane.buildApi.port | int | `8001` | Build API port (git proxy and token validation) |
| controlPlane.buildApi.metricsPath | string | `"/metrics"` | Metrics path for the build API |
| controlPlane.hpa.enabled | bool | `false` | Enable HPA for the control plane |
| controlPlane.hpa.minReplicas | int | `1` | Minimum replicas |
| controlPlane.hpa.maxReplicas | int | `3` | Maximum replicas |
| controlPlane.hpa.targetCPUUtilizationPercentage | int | `80` | Target average CPU utilization percentage |

### Object Storage

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| controlPlane.objectStorage.s3.endpointUrl | string | `""` | S3 endpoint URL (leave empty for AWS) |
| controlPlane.objectStorage.s3.bucket | string | `""` | S3 bucket name (**required**) |
| controlPlane.objectStorage.s3.region | string | `""` | S3 region |
| controlPlane.objectStorage.s3.unsigned | bool | `false` | Send S3 requests unsigned (no Authorization header). Enable for authless backends like s3proxy/LocalStack or public-read buckets. Leave `false` for real AWS, MinIO, or any auth-requiring backend. |
| controlPlane.objectStorage.secretRef | string | `""` | K8s Secret name containing `S3_ACCESS_KEY` and `S3_SECRET_KEY` |
| controlPlane.objectStorage.buildKeyPrefix | string | `"builds"` | Key prefix for build artifacts in the bucket |
| controlPlane.objectStorage.backupKeyPrefix | string | `"backups"` | Key prefix for backup archives in the bucket |
| controlPlane.objectStorage.codeRepoKeyPrefix | string | `"git"` | Key prefix for code repositories in the bucket |
| controlPlane.objectStorage.backupEncryptionSecretRef | string | `""` | K8s Secret name containing `BACKUP_ENCRYPTION_PASSWORD` |

### Apps

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| apps.namespace | string | `""` | Namespace where LlamaDeployment CRs and all operator-managed child resources live. Empty = release namespace. When set, the operator + control plane stay in the release namespace and target this namespace for all app resources. |

### CRDs

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| crds.version | string | `"0.7.2"` | Compatible `llama-agents-crds` chart version for this release. Documentation only; not read by templates. Auto-synced at release time. |

### Operator

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| operator.enabled | bool | `true` | Deploy the operator |
| operator.replicas | int | `1` | Number of operator replicas |
| operator.annotations | object | `{}` | Annotations for the operator Deployment |
| operator.podAnnotations | object | `{}` | Annotations for the operator pod template |
| operator.defaultAppRequests.cpu | string | `"750m"` | Default CPU request for managed app containers |
| operator.defaultAppRequests.memory | string | `"2Gi"` | Default memory request for managed app containers |
| operator.defaultAppLimits.cpu | string | `""` | Default CPU limit for managed app containers (empty = no limit) |
| operator.defaultAppLimits.memory | string | `"4096Mi"` | Default memory limit for managed app containers |
| operator.resources | object | `{limits: {cpu: 500m, memory: 128Mi}, requests: {cpu: 10m, memory: 64Mi}}` | Resource requests/limits for the operator container |
| operator.maxConcurrentRollouts | int | `10` | Max simultaneous LlamaDeployment rollouts (0 = unlimited) |
| operator.maxDeployments | int | `0` | Max active LlamaDeployments per namespace (0 = unlimited) |
| operator.env | list | `[]` | Extra environment variables for the operator container |
| operator.rolloutTimeoutSeconds | int | `1800` | Rollout timeout in seconds for managed deployments |
| operator.llamaDeploymentTemplate.enabled | bool | `false` | Create a default LlamaDeploymentTemplate in the namespace |
| operator.llamaDeploymentTemplate.name | string | `"default"` | Template resource name |
| operator.llamaDeploymentTemplate.metadata | object | `{}` | Metadata for the template (labels, annotations) |
| operator.llamaDeploymentTemplate.spec | object | `{"podSpec":{}}` | Template spec (podSpec with nodeSelector, tolerations, affinity, container overrides) |
| operator.hpa.enabled | bool | `false` | Enable HPA for the operator |
| operator.hpa.minReplicas | int | `1` | Minimum replicas |
| operator.hpa.maxReplicas | int | `3` | Maximum replicas |
| operator.hpa.targetCPUUtilizationPercentage | int | `80` | Target average CPU utilization percentage |

### Local Development

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| localDev.enabled | bool | `false` | Enable local dev ingress for deployed apps |
| localDev.ingressDomain | string | `"127.0.0.1.nip.io"` | Ingress domain for local dev |

### RBAC

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| rbac.create | bool | `true` | Create Role and RoleBinding |
| rbac.roleAnnotations | object | `{}` | Annotations for the Role |
| rbac.roleBindingAnnotations | object | `{}` | Annotations for the RoleBinding |

### Service Account

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| serviceAccount.create | bool | `true` | Create a ServiceAccount |
| serviceAccount.name | string | `"llama-agents"` | ServiceAccount name |
| serviceAccount.annotations | object | `{}` | Annotations for the ServiceAccount |

### Network Policy

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| networkPolicy.enabled | bool | `true` | Enable egress NetworkPolicy for operator-managed pods |
| networkPolicy.extraMatchExpressions | list | `[]` | Additional pod selector matchExpressions |
| networkPolicy.extraEgressRules | list | `[]` | Extra egress rules appended to the NetworkPolicy |
| networkPolicy.blockPrivateRanges | bool | `true` | Block private IP ranges (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16) in internet egress rule |
| networkPolicy.dns.namespaceSelector | object | `{"kubernetes.io/metadata.name":"kube-system"}` | Namespace selector for DNS pods. Defaults to kube-system |
| networkPolicy.dns.podSelector | object | `{"k8s-app":"kube-dns"}` | Pod selector for DNS pods. Defaults to kube-dns |

## Uninstalling

```bash
helm uninstall llama-agents
```

CRDs are **not** removed on uninstall. To remove them (this deletes all LlamaDeployment resources):

```bash
kubectl delete crd llamadeployments.deploy.llamaindex.ai llamadeploymenttemplates.deploy.llamaindex.ai
```
