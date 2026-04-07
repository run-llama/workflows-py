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
