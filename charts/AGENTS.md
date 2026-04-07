## Helm Chart

Makefile lives in `operator/`. Run targets with `make -C operator <target>` from repo root, or `make <target>` from within `operator/`.

Setup:
- Ensure kind context: `make -C operator kube-ensure-kind-context`
- Install helm-unittest: `make -C operator helm-unittest-install`
- Install Prometheus Operator CRDs: `make -C operator helm-crds-prom-operator`

Checks:
- Lint (default values): `make -C operator helm-lint`
- Lint (dev values): `make -C operator helm-lint-dev`
- Template (default/dev): `make -C operator helm-template` / `make -C operator helm-template-dev`
- Server-side dry-run (default/dev): `make -C operator helm-dry-run` / `make -C operator helm-dry-run-dev`
- Run Helm unit tests: `make -C operator helm-unittest`
