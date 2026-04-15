# Operator Apps Namespace Mode

## Motivation

Today the operator, the control plane, and every app they deploy all live in
a single namespace. The operator's `WATCH_NAMESPACE` env var (see
`operator/cmd/main.go:193`) configures the namespace it watches for
`LlamaDeployment` CRs, which also happens to be the namespace where it
creates child resources.

We want an additional mode where:

- The operator + control plane run in a "system" namespace (e.g.
  `llama-agents`).
- `LlamaDeployment` CRs **and** their child resources (Deployments, Pods,
  Services, Jobs, Secrets, ServiceAccounts, ConfigMaps, Ingresses,
  NetworkPolicies) all live in a separate **apps namespace** (e.g.
  `llama-agents-apps`).
- That apps namespace can have tighter quotas, stricter network policies,
  different node pools, and separate audit/compliance boundaries from the
  system components.

Crucially, CRs live **with** their child resources. Cross-namespace owner
references are illegal in Kubernetes, so co-locating the CR with its pods
keeps `SetControllerReference` working, keeps garbage collection working,
and keeps the reconciler's ownership model unchanged.

## Goals

1. New Helm value `operator.apps.namespace` (string, default empty = same as
   release namespace). When set, it targets the operator's `WATCH_NAMESPACE`
   and the control plane's `KUBERNETES_NAMESPACE` at that namespace.
2. No changes to the reconciler's ownership, finalizer, or child-resource
   construction logic. Those already use `ld.Namespace`; pointing the
   operator at a different namespace is enough.
3. Helm chart provisions RBAC in the apps namespace and a minimal RBAC in
   the release namespace (leader-election Lease only).
4. Per-app `NetworkPolicy` keeps working when app pods and control plane
   live in different namespaces.
5. No behaviour change for existing users who do not set
   `operator.apps.namespace`.

## Non-Goals

- **Multiple apps namespaces.** A future extension may add
  `operator.apps.namespaces` (plural list) alongside the singular form.
  This plan intentionally ships the singular variant only — partial
  plural support is confusing and easy to get wrong.
- **Per-CR target namespace (`spec.targetNamespace`).** Not supported.
- **Cluster-wide operator.** Still a single watched namespace.
- **Live migration** of existing deployments across namespaces. Switching
  modes requires recreating `LlamaDeployment` CRs.

## Architecture Summary

```
                 ┌────────────────────────────────┐
                 │ system / release namespace     │
                 │                                │
                 │ - operator pod + SA            │
                 │ - control plane + build API    │
                 │ - leader-election Lease        │
                 │ - operator/control-plane       │
                 │   Services                     │
                 └────────────┬───────────────────┘
                              │ reconciles / serves API
                              ▼
                 ┌────────────────────────────────┐
                 │ apps namespace                 │
                 │  (operator.apps.namespace)     │
                 │                                │
                 │ - LlamaDeployment CRs          │
                 │ - LlamaDeploymentTemplate CRs  │
                 │ - app Deployment + Pods        │
                 │ - Service / Ingress            │
                 │ - NetworkPolicy (per-app)      │
                 │ - ConfigMap (nginx)            │
                 │ - Secret (per-CR)              │
                 │ - ServiceAccount (per-CR)      │
                 │ - build Job + Pods             │
                 │ - CR + child Events            │
                 └────────────────────────────────┘
```

Because CRs and their children share a namespace, standard owner references
work and the reconciler's existing behaviour (`.Owns(...)`, finalizer,
`SetControllerReference`, capacity gates scoped by `current.Namespace`)
all apply unchanged.

## Design

### 1. Helm chart (`charts/llama-agents/`)

New value shape (object form, leaves room for a later plural field):

```yaml
operator:
  apps:
    # Namespace for LlamaDeployment CRs and their child resources.
    # Empty = same as the release namespace (legacy behaviour).
    namespace: ""
```

Template updates:

- `_helpers.tpl`: add `llama-agents.apps.namespace` that returns
  `.Values.operator.apps.namespace` if non-empty, otherwise
  `.Release.Namespace`.
- `templates/operator-deployment.yaml:52`: `WATCH_NAMESPACE` is no longer
  sourced from `metadata.namespace`. Set it explicitly to
  `{{ include "llama-agents.apps.namespace" . }}`.
- `templates/deployment.yaml` (control plane): add / override
  `KUBERNETES_NAMESPACE={{ include "llama-agents.apps.namespace" . }}` on
  the control plane + build API containers.
- `templates/rbac.yaml`: today this emits one Role + RoleBinding in
  `.Release.Namespace`. Change to:
  - **Apps-namespace Role + RoleBinding** (namespace =
    `llama-agents.apps.namespace`) carrying every rule currently in the
    template *except* `coordination.k8s.io/leases`. Binds to the operator
    SA in the release namespace.
  - **Release-namespace Role + RoleBinding** (namespace =
    `.Release.Namespace`) with only `coordination.k8s.io/leases` for
    leader election. Binds to the same SA.
  - When the two namespaces resolve to the same value (legacy mode),
    collapse back into a single combined Role + RoleBinding in the
    release namespace, so small/default installs don't get extra objects.
- `templates/networkpolicy.yaml`: two changes:
  - Set `metadata.namespace: {{ include "llama-agents.apps.namespace" . }}`
    (the NP must live where the pods it selects live).
  - The egress rule at lines 23-29 (`- to: podSelector: { app: controlplane }`)
    currently relies on implicit same-namespace selection. Add a
    `namespaceSelector` to allow reaching the control plane when the
    release namespace differs:
    ```yaml
    - to:
      - namespaceSelector:
          matchLabels:
            kubernetes.io/metadata.name: {{ .Release.Namespace }}
        podSelector:
          matchLabels:
            app: {{ include "llama-agents.controlplane.name" . }}
    ```
- The operator's `LLAMA_DEPLOY_BUILD_API_HOST` env at
  `operator-deployment.yaml:91` already qualifies the service FQDN with
  `{{ .Release.Namespace }}`, so build Pods in the apps namespace reach
  the build API fine. No change.

Add chart snapshot / `helm template` smoke coverage for both the empty
and set cases: assert the `WATCH_NAMESPACE`, `KUBERNETES_NAMESPACE`, Role
namespaces, and NetworkPolicy namespace selector render correctly.

### 2. Operator (`operator/cmd/main.go`, controller package)

**No functional code changes expected.** The operator already reads
`WATCH_NAMESPACE` and uses it as the sole cache-restricted namespace
(`cmd/main.go:193-214`). Pointing it at a namespace the operator pod does
not run in is already supported by controller-runtime.

Leader-election Lease placement: controller-runtime's default is the pod's
namespace (derived from the service account token mount). That's the
release namespace — exactly where we want it. No `LeaderElectionNamespace`
override needed.

Verification work (not code changes):

- Confirm the direct (non-cached) client at `cmd/main.go:236` honors the
  apps-namespace RBAC when resolving ReplicaSets (it's created with
  `ctrl.GetConfigOrDie()`, same credentials as the cached client —
  expected to work).
- Confirm the Events recorder writes Events into the apps namespace
  (it writes to `involvedObject.Namespace`, which for the CR is the apps
  namespace — expected to work).
- Confirm that `LlamaDeploymentTemplate` lookup
  (`resources.go:668`, `resources.go:1287`) resolves correctly when the
  template CR is created in the apps namespace.

### 3. Control plane

The control plane already has `KUBERNETES_NAMESPACE` (see
`packages/llama-agents-control-plane/src/llama_agents/control_plane/settings.py:15`).
All CR creates/reads, Secret creates/reads, pod/log lookups, service
lookups, and ingress management go through `self.namespace` in
`K8sClient`, which is set from this env var.

Only change: the Helm chart sets this env var to the apps namespace (see
§1). No Python code changes.

### 4. NetworkPolicy (operator-generated, per-app)

Out of scope for the chart template above — the operator also generates a
per-deployment NetworkPolicy inside `resources.go`. Audit it:

- If it references "same namespace" implicit selectors (podSelector only,
  no namespaceSelector), it will still work because both the NP and the
  app pods live in the apps namespace.
- If it references the control plane via `podSelector` (no
  `namespaceSelector`), add a `namespaceSelector` pointing at the release
  namespace. The release namespace must be readable by the operator at
  render time — plumb it through as an env var
  (`LLAMA_DEPLOY_SYSTEM_NAMESPACE`) set from the chart:
  ```yaml
  - name: LLAMA_DEPLOY_SYSTEM_NAMESPACE
    valueFrom:
      fieldRef:
        fieldPath: metadata.namespace
  ```
- The reconciler reads this env var and, when non-empty *and* different
  from the CR's namespace, emits the cross-namespace rules. Default to
  same-namespace rules (today's behaviour).

### 5. ImagePullSecrets

If users rely on imagePullSecrets for the appserver image, those secrets
currently live in the release namespace and won't be visible to pods in
the apps namespace. Options for users (documented, not enforced by the
chart):

- Create the pull secret in the apps namespace themselves.
- Or use a node-level imagePullSecret via the node's kubelet config.

The chart does not mirror or manage this secret; mention it in the Helm
README and the architecture doc.

### 6. Dev environment (`operator/dev.py`, `operator/tilt/`)

Tilt and `dev.py` currently create everything in one namespace. Options:

- Leave as-is; local dev runs in legacy (single-namespace) mode.
- Add a `--apps-namespace` flag to `dev.py` that flips the Helm value.

Recommendation: ship legacy-only local dev for this PR. Document that
`operator.apps.namespace` is tested via envtest + a Helm template test,
and that Tilt dev is single-namespace for now.

### 7. Tests

**Chart (`charts/llama-agents/tests/` or similar):** add Helm template
assertions for two cases:

- `operator.apps.namespace: ""` → single combined Role in release
  namespace, NetworkPolicy in release namespace, `WATCH_NAMESPACE` points
  at release namespace.
- `operator.apps.namespace: "llama-agents-apps"` → two split Roles,
  NetworkPolicy in apps namespace with the cross-namespace selector,
  `WATCH_NAMESPACE` and `KUBERNETES_NAMESPACE` point at apps namespace.

**Operator integration test (`//go:build integration`):** extend
`suite_test.go` / `llamadeployment_controller_test.go` to create an
additional namespace, run the reconciler scoped to that namespace, and
assert a `LlamaDeployment` created there reconciles normally and gets
cleaned up on deletion.

**Operator unit test:** if we add the `LLAMA_DEPLOY_SYSTEM_NAMESPACE`
plumbing for per-app NetworkPolicy (§4), unit-test both the same-ns and
split-ns rendering.

No new control-plane tests required (no code change).

### 8. Docs + changeset

- Update `architecture-docs/overall-architecture.md` with the namespace
  split and note that CRs live with their pods.
- Append to `operator/AGENTS.md`: document
  `operator.apps.namespace`, the two-Role RBAC, and how to run the
  integration test.
- Helm README: the usage section must explain the `imagePullSecrets`
  caveat.
- Changeset entry: "Operator supports a dedicated namespace for
  LlamaDeployment CRs and their app resources."

## Rollout Phases

Each phase ends with `uv run dev`, `uv run pre-commit run -a`,
`make -C operator operator-unit-test`, and `make -C operator
operator-test`.

1. **Chart plumbing**: add `operator.apps.namespace`, the helper, the
   `WATCH_NAMESPACE` / `KUBERNETES_NAMESPACE` wiring, and the split RBAC
   with fallback-to-single-role. Helm template tests.
2. **Per-app NetworkPolicy**: plumb `LLAMA_DEPLOY_SYSTEM_NAMESPACE` into
   the operator and emit cross-namespace selectors when the two
   namespaces differ. Unit tests.
3. **Integration test**: envtest covering cross-namespace reconciliation.
4. **Docs + changeset**.

## Open Questions

- Do we want a namespace-label validator (e.g. require
  `pod-security.kubernetes.io/enforce=restricted` on the apps namespace)?
  Out of scope; document recommended labels instead.
- Local dev parity — worth adding an apps-namespace mode to `dev.py`?
  Defer until someone is actively debugging the split config locally.
- Build Jobs share the apps namespace. Acceptable for now; a separate
  `*-builds` namespace is a future optimisation if build workloads grow
  noisy.

## Risk / Compatibility Matrix

| Risk | Mitigation |
| --- | --- |
| Existing installs unaware of new value get different behaviour | Empty `operator.apps.namespace` resolves to release namespace; RBAC collapses to today's single Role. No observable change. |
| Switching modes on a live install orphans in-place apps | Documented as destructive: drain + recreate `LlamaDeployment` CRs. |
| Per-app NetworkPolicy drops control-plane traffic in split mode | §4: emit cross-namespace `namespaceSelector` when `LLAMA_DEPLOY_SYSTEM_NAMESPACE` is set and differs from the CR ns. |
| imagePullSecrets not present in apps namespace | Documented; user responsibility. |
| Leader election Lease ends up in wrong namespace | controller-runtime defaults to the pod's namespace (release ns). Keep that default. |
| Future plural support breaks the singular value | The singular sits under `operator.apps.namespace`; a future plural
`operator.apps.namespaces` can coexist with a precedence rule ("if
plural is set, singular is ignored"). |

## Forward Compatibility Note

This plan uses `operator.apps.namespace` (singular object field) rather
than something like `operator.appsNamespace` (flat singular) or
`operator.apps.namespaces` (plural list) for one reason: if and when
plural support lands, it can be added as a sibling field under
`operator.apps` without renaming or deprecating the singular, and without
misleading users today about capabilities that don't exist yet.
