# Operator Target Namespace Mode

## Motivation

Today the operator, the control plane, and every app it deploys all live in a
single namespace. The operator's `WATCH_NAMESPACE` env var (see
`operator/cmd/main.go:193`) configures both the namespace it watches for
`LlamaDeployment` CRs **and** the namespace where it creates child resources
(Deployments, Services, Jobs, Secrets, Ingresses, NetworkPolicies,
ServiceAccounts, ConfigMaps).

We want an additional mode where:

- The operator + control plane run in a "system" namespace (e.g. `llama-agents`).
- `LlamaDeployment` CRs are still reconciled from the system namespace.
- Child resources for each app are created in a separate **target namespace**
  (e.g. `llama-agents-apps`) whose policies (quotas, network policies, PSPs,
  admission controllers, node pools) can be tightened independently.

This is the foundation for running the system components with fewer privileges
than the app payloads, and for keeping the app namespace's object churn from
bloating the informer caches used by the control plane.

## Goals

1. New operator config: `LLAMA_DEPLOY_TARGET_NAMESPACE` env / `--target-namespace`
   flag. Empty = disabled (today's behaviour, no regression).
2. When enabled:
   - `LlamaDeployment` / `LlamaDeploymentTemplate` CRs continue to live in
     `WATCH_NAMESPACE`.
   - All child resources are created in the target namespace.
   - Operator cache watches both namespaces, with type-scoped selectors to
     keep informer memory bounded.
3. Helm chart support:
   - New value `operator.targetNamespace`.
   - RBAC split across the two namespaces.
   - Optional creation of the target namespace by the chart.
4. Control plane updates so pod/log discovery (used for run status + log
   streaming) looks in the target namespace.
5. No change to behaviour for existing users who do not opt in.

## Non-Goals

- Per-`LlamaDeployment` target namespace (`spec.targetNamespace`). Deferred;
  the global mode is a prerequisite and keeps the blast radius small.
- Cluster-wide operator (watching all namespaces). Still a single
  `WATCH_NAMESPACE`, single target namespace.
- Migration of existing in-place child resources across namespaces. Switching
  the mode requires re-creating the deployments; we document this rather than
  attempt a live migration.

## Architecture Summary

```
                 ┌────────────────────────────────┐
                 │ system namespace               │
                 │  (WATCH_NAMESPACE)             │
                 │                                │
                 │ - operator pod                 │
                 │ - control plane + build API    │
                 │ - LlamaDeployment CRs          │
                 │ - LlamaDeploymentTemplate CRs  │
                 │ - user-provided Secret (src)   │
                 │ - Events, Leases               │
                 └────────────┬───────────────────┘
                              │ reconciles
                              ▼
                 ┌────────────────────────────────┐
                 │ target namespace               │
                 │  (LLAMA_DEPLOY_TARGET_NAMESPACE)│
                 │                                │
                 │ - app Deployment + Pods        │
                 │ - Service / Ingress            │
                 │ - ConfigMap (nginx)            │
                 │ - mirrored Secret (app copy)   │
                 │ - ServiceAccount               │
                 │ - NetworkPolicy                │
                 │ - build Job + Pods             │
                 └────────────────────────────────┘
```

Cross-namespace `ownerReferences` are **not** supported by Kubernetes (the
owner must live in the same namespace as the dependent). We therefore switch
to label-based ownership + finalizer-driven cleanup for target-namespace
mode, and keep `SetControllerReference` only for children that live in the
CR's own namespace.

## Design

### 1. Operator config (`operator/cmd/main.go`)

- Add `config.TargetNamespace`, parsed from env var
  `LLAMA_DEPLOY_TARGET_NAMESPACE` with an optional
  `--target-namespace` flag that overrides it.
- Pass `TargetNamespace` into `LlamaDeploymentReconciler`.
- Cache configuration:
  - If target namespace is empty → existing single-namespace cache config.
  - If non-empty → `cache.Options.DefaultNamespaces` has both
    `watchNamespace` and `targetNamespace` keys, **plus** per-type
    restrictions in `ByObject`:
    - `LlamaDeployment`, `LlamaDeploymentTemplate`, `Event`, `Lease`,
      user-provided `Secret`: watched only in `watchNamespace`.
    - `Deployment`, `ReplicaSet`, `Pod`, `Service`, `Ingress`,
      `NetworkPolicy`, `ConfigMap`, `ServiceAccount`, `Job`, mirrored
      `Secret`: watched only in `targetNamespace`.
  - This is important: a naked `DefaultNamespaces` map would cause the
    operator to watch Pods in the system namespace too, which we want to
    avoid for memory reasons.

### 2. Reconciler plumbing (`operator/internal/controller/`)

Add fields and helpers:

```go
type LlamaDeploymentReconciler struct {
    // ... existing fields ...
    // TargetNamespace is the namespace where child resources (Deployments,
    // Jobs, Services, etc.) are created. Empty means "same as the CR's
    // own namespace" (legacy mode).
    TargetNamespace string
}

// targetNamespaceFor returns the namespace where child resources for the
// given LlamaDeployment should live.
func (r *LlamaDeploymentReconciler) targetNamespaceFor(ld *llamadeployv1.LlamaDeployment) string {
    if r.TargetNamespace != "" {
        return r.TargetNamespace
    }
    return ld.Namespace
}

// isSplitNamespace reports whether CR and children live in different namespaces.
func (r *LlamaDeploymentReconciler) isSplitNamespace(ld *llamadeployv1.LlamaDeployment) bool {
    return r.TargetNamespace != "" && r.TargetNamespace != ld.Namespace
}
```

Replace every `llamaDeploy.Namespace` used when constructing or fetching a
**child** resource with `r.targetNamespaceFor(llamaDeploy)`. Concrete sites
to update in `operator/internal/controller/resources.go` (see grep hits at
lines 396, 464, 606, 806, 849, 870, 895, 917, 936, 1103 plus similar in
`lifecycle.go`).

Keep `ld.Namespace` for:

- Listing sibling `LlamaDeployment`s for capacity gates
  (`lifecycle.go:47`, `lifecycle.go:91`).
- Reading the user-provided Secret referenced by `spec.secretName` — source
  lives with the CR (`resources.go:849`).
- Looking up `LlamaDeploymentTemplate` (`resources.go:668`,
  `resources.go:1287`).
- Updating the CR's own status.

### 3. Ownership + cleanup

Cross-namespace owner references are rejected by the API server, so:

- In `resources.go`, wrap every `controllerutil.SetControllerReference(ld,
  child, ...)` with a guard: only call it if
  `child.GetNamespace() == ld.Namespace`.
- Ensure every child resource carries stable labels regardless of mode:
  - `app.kubernetes.io/managed-by: llama-deploy-operator`
  - `deploy.llamaindex.ai/deployment: <ld.Name>`
  - `deploy.llamaindex.ai/cr-namespace: <ld.Namespace>` **(new)** — used by
    the finalizer to identify which CR owns a child in the target namespace.
- Rewrite watch setup in `SetupWithManager`:
  - Replace `.Owns(&appsv1.Deployment{})` / `.Owns(&batchv1.Job{})` etc.
    with `.Watches(&appsv1.Deployment{}, handler.EnqueueRequestsFromMapFunc(r.mapChildToCR))`.
  - `mapChildToCR` reads the two labels above and emits a
    `reconcile.Request{Namespace: cr-namespace, Name: deployment-label}`.
  - Keep `.Owns` for anything still created in the CR namespace (none in
    split mode, but the mapper works for both modes so we can just use
    watches everywhere).
- Finalizer deletion (`handleDeletion` in `lifecycle.go`): today relies on
  owner-ref garbage collection. In split mode we must explicitly list and
  delete children in the target namespace using label selectors:
  `deploy.llamaindex.ai/deployment=<ld.Name>,deploy.llamaindex.ai/cr-namespace=<ld.Namespace>`.
  - Delete order: Deployments → ReplicaSets (orphan) → Services → Ingresses
    → NetworkPolicies → Jobs → ConfigMaps → Secrets → ServiceAccounts.
  - Block finalizer removal until no labelled children remain (to avoid
    orphan resources).

### 4. Secret mirroring

The app pod's `envFrom: secretRef` must resolve to a Secret in the same
namespace as the pod (target namespace), but the user supplies the Secret in
the system namespace via the control plane.

Add a `reconcileMirroredSecret` step in `resources.go`:

- Read the source Secret from `ld.Namespace` (existing
  `checkSecretGate` logic stays).
- Maintain a child Secret in the target namespace named
  `<ld.Name>-secret` (or equivalent). Copy `Data` and `StringData` 1:1.
- Labels as above + an annotation
  `deploy.llamaindex.ai/source-secret: <ld.Namespace>/<spec.secretName>`
  for traceability.
- Watch the source Secret via the existing watch on `watchNamespace` — a
  change triggers reconciliation, which updates the mirror.
- App Deployment's `envFrom` is changed to reference the mirrored secret
  when split mode is active (still `spec.secretName` in legacy mode).

### 5. Service account

Today each `LlamaDeployment` gets its own SA (`<name>-sa`) in the same
namespace as the pod (`resources.go:806`). With split namespaces the SA
must be created in the target namespace. No change needed beyond using
`r.targetNamespaceFor(ld)`.

### 6. Control plane changes

File: `packages/llama-agents-control-plane/src/llama_agents/control_plane/`.

- `settings.py`: add `kubernetes_target_namespace` field (alias
  `KUBERNETES_TARGET_NAMESPACE`), defaulting to empty = "same as
  `kubernetes_namespace`".
- `k8s_client.py`: split into `cr_namespace` (existing `self.namespace`)
  and `target_namespace`. All pod / log / service / ingress queries use
  `target_namespace`; all `LlamaDeployment` CR operations use
  `cr_namespace`.
- Helm chart: plumb `operator.targetNamespace` into the control plane's
  `KUBERNETES_TARGET_NAMESPACE` env var too (both the deployment and the
  build API container).

### 7. Helm chart (`charts/llama-agents/`)

New values:

```yaml
operator:
  # Empty means "deploy apps into the same namespace as the operator".
  targetNamespace: ""
  # If true, chart creates the Namespace resource (useful when installed
  # by a cluster admin).
  createTargetNamespace: false
```

Template updates:

- `templates/operator-deployment.yaml`: add env
  ```yaml
  - name: LLAMA_DEPLOY_TARGET_NAMESPACE
    value: {{ .Values.operator.targetNamespace | quote }}
  ```
- `templates/rbac.yaml`: split into two Roles + two RoleBindings when
  `operator.targetNamespace` is set and not equal to `.Release.Namespace`:
  - **System Role** (release namespace): `llamadeployments*`,
    `llamadeploymenttemplates`, `events`, `leases`, `secrets` (get/list/watch
    only, for reading user secrets).
  - **Target Role** (target namespace): `deployments`, `replicasets`,
    `services`, `ingresses`, `networkpolicies`, `configmaps`, `secrets`
    (full), `serviceaccounts`, `pods`, `pods/log`, `jobs`, `events`.
  - Each RoleBinding points at the operator ServiceAccount in the release
    namespace.
- New optional template `templates/target-namespace.yaml` (only rendered
  when `createTargetNamespace: true`) that emits a `Namespace` resource
  with matching labels.
- `templates/networkpolicy.yaml`: the existing per-app NetworkPolicy is
  already namespace-local; confirm it still works when emitted in the
  target namespace.
- `templates/controlplane*.yaml`: add
  `KUBERNETES_TARGET_NAMESPACE={{ .Values.operator.targetNamespace }}`
  to the control plane container env.

Keep `operator/config/rbac/role.yaml` (the kubebuilder-generated default)
alone — it's the cluster-scoped reference RBAC used in the `make
operator-manifests` output. Document that the Helm chart is the source of
truth for runtime RBAC.

### 8. Tests

Unit (`operator/internal/controller/`):

- `resources_namespace_test.go` (new, `//go:build !integration`):
  - With `TargetNamespace = ""`, child resources land in CR namespace
    (baseline).
  - With `TargetNamespace = "apps"`, Deployment/Service/Ingress/Job/SA
    land in `apps` and carry the `cr-namespace` label.
  - `SetControllerReference` is **not** called when namespaces differ.
- `lifecycle_namespace_test.go`: finalizer cleanup deletes labelled
  children in the target namespace and blocks finalizer removal while any
  remain.
- `resources_secret_mirror_test.go`: source secret in CR namespace; mirror
  appears in target namespace with matching Data; updating source updates
  mirror.

Integration (`//go:build integration`):

- New test in `llamadeployment_controller_test.go` or a sibling file using
  envtest:
  - Two namespaces pre-created by the suite.
  - Reconciler constructed with `TargetNamespace` set.
  - Create a `LlamaDeployment`, assert Deployment/Service appear in target
    namespace, and that CR deletion removes them.

Update `test_utils_test.go`:

- `NewTestReconciler` accepts an optional `WithTargetNamespace(ns string)`
  functional option.
- Envtest suite in `suite_test.go` creates the extra namespace when a test
  requests it.

Control plane:

- Unit test for `settings.py` / `k8s_client.py` confirming
  `target_namespace` override and fallback behaviour.
- Existing tests (`test_find_deployment_id.py` etc.) updated where they
  assume a single namespace.

Chart:

- Add snapshot-style tests in `charts/llama-agents/tests/` exercising
  `operator.targetNamespace` (both empty and set), asserting the presence
  of the split Role / RoleBinding and the extra env var.

### 9. Docs + changeset

- Update `architecture-docs/overall-architecture.md` with the new
  namespace split and a short section on label-based ownership.
- Append a note in `operator/AGENTS.md` describing
  `LLAMA_DEPLOY_TARGET_NAMESPACE` and how to run envtest with the second
  namespace.
- Add `npx changeset` entry: "Operator supports a separate target
  namespace for app deployments."

## Rollout Phases

Each phase ends with `uv run dev`, `uv run pre-commit run -a`, and
`make -C operator operator-unit-test` (plus `operator-test` for the envtest
integration phase).

1. **Plumbing**: add `TargetNamespace` field, `targetNamespaceFor` helper,
   env/flag wiring, and cache config. All existing callers still pass the
   CR namespace explicitly; no functional change yet.
2. **Child resources**: switch every child-resource construction and
   lookup site to `r.targetNamespaceFor(ld)`; guard `SetControllerReference`
   on same-namespace; add labels; replace `.Owns` with `.Watches`.
3. **Finalizer cleanup**: add label-based deletion walker in
   `handleDeletion`.
4. **Secret mirroring**: add `reconcileMirroredSecret` and flip
   `envFrom` to mirror in split mode.
5. **Control plane**: add `target_namespace` setting and route pod/log
   queries through it.
6. **Helm chart**: values, env plumbing, split RBAC, optional namespace
   creation, tests.
7. **Docs + changeset**.

## Open Questions

- Should we let each `LlamaDeploymentTemplate` override the target
  namespace? Probably no for the first iteration — keep the global knob
  until we have a clear use case.
- Build Jobs can be chatty; do we want them in the target namespace too,
  or in a dedicated `*-builds` namespace? Plan keeps them in the target
  namespace for simplicity. A future follow-up could split them out behind
  another env var.
- User-provided Secret lifecycle: do we delete the mirrored copy eagerly
  when the source is deleted? Plan says yes — the source Secret being
  missing flips the CR into `AwaitingCode` / a failure state, at which
  point the reconciler can drop the mirror.
- Do we need a ValidatingAdmissionWebhook to reject CRs in namespaces
  other than the configured `WATCH_NAMESPACE`? Out of scope; the cache
  already filters them out, so such CRs simply go unreconciled.

## Risk / Compatibility Matrix

| Risk | Mitigation |
| --- | --- |
| Existing users unaware of new env var get different behaviour | Empty `TargetNamespace` preserves legacy path verbatim. |
| Informer memory blows up watching Pods in two namespaces | Per-type `ByObject` namespace selectors; Pods only watched in target. |
| Stale children left behind after CR deletion | Finalizer explicitly walks target namespace via labels before clearing itself. |
| Secret drift between source and mirror | Mirror reconciled on every pass; source Secret watched for change events. |
| Operator can't set owner refs → orphans on crash-deletion | Label selectors + finalizer. A `kubectl delete ns` on the system namespace without draining CRs still leaks target-namespace children; document this as a known limitation. |
| Switching modes on a live system breaks in-place apps | Documented as destructive: drain + recreate LlamaDeployments. |
