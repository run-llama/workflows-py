## Operator (Go)

Makefile lives in `operator/`. Run targets with `make -C operator <target>` from repo root, or `make <target>` from within `operator/`.

Setup:
- Generate CRDs/RBAC: `make -C operator operator-manifests`
- Generate deepcopy: `make -C operator operator-generate`

Checks (run all before pushing):
- Lint: `make -C operator operator-lint`
- Unit tests (fast, no envtest): `make -C operator operator-unit-test`
- Integration tests (envtest): `make -C operator operator-test`

Running specific tests:
- Unit: `cd operator && go test -tags='!integration' ./internal/controller/ -run TestMyTest -v`
- Integration: `eval "$($(go env GOPATH)/bin/setup-envtest use -p env)" && cd operator && go test -tags=integration ./internal/controller/ -run 'TestControllers/MyTest' -v`
- Unit tests use `//go:build !integration`, integration tests use `//go:build integration`

Test patterns:
- Unit tests use Go `testing` + fake client (`sigs.k8s.io/controller-runtime/pkg/client/fake`)
- Integration tests use Ginkgo/Gomega + envtest (real API server). Test names are `TestControllers/description`.
- Shared helpers are in `test_utils_test.go` (e.g. `NewTestReconciler`, `NewLlama`, `CreateAndReconcile`, `CompleteBuild`)

Local run:
- `make -C operator operator-run` (requires kubeconfig)

### Apps namespace mode

The Helm chart supports an optional "apps namespace" split via
`operator.apps.namespace` — when set, `LlamaDeployment` CRs and all
operator-managed child resources live in that namespace while the operator +
control plane stay in the release namespace. The operator itself is unchanged;
only its `WATCH_NAMESPACE` (and the control plane's `KUBERNETES_NAMESPACE`)
move. See `architecture-docs/overall-architecture.md` for the full layout.

Integration coverage for this mode lives in
`internal/controller/apps_namespace_test.go` — it reconciles a CR in a
non-default namespace and asserts children land there with working owner
references. Run with `make -C operator operator-test`.
