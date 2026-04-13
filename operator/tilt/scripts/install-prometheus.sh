#!/usr/bin/env bash
# Installs a minimal prometheus-operator + Prometheus instance for local dev.
# Idempotent — safe to re-run.
set -euo pipefail

PROM_OP_VERSION="v0.89.0"
NAMESPACE="monitoring"
CONTEXT="${1:-kind-kind}"
KUBECTL="kubectl --context=$CONTEXT"

# Create namespace if needed
$KUBECTL get namespace "$NAMESPACE" &>/dev/null || \
  $KUBECTL create namespace "$NAMESPACE"

# Install prometheus-operator bundle (CRDs + RBAC + Deployment).
# The upstream bundle targets the default namespace — rewrite to monitoring.
curl -sL "https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/${PROM_OP_VERSION}/bundle.yaml" \
  | sed "s/namespace: default/namespace: ${NAMESPACE}/g" \
  | $KUBECTL apply --server-side -f - -n "$NAMESPACE"

# Wait for the operator to be ready before creating the Prometheus CR
$KUBECTL rollout status deployment/prometheus-operator -n "$NAMESPACE" --timeout=120s

# Apply the Prometheus instance
$KUBECTL apply -f "$(dirname "$0")/../k8s-manifests/prometheus.yaml"
