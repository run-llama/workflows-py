#!/usr/bin/env python3
"""
Process generated manifests from kubebuilder and create Helm templates.
This preserves Helm templating while keeping kubebuilder annotations as source of truth.
"""

import sys
from pathlib import Path

import yaml

# Resolve repo root relative to this script's location (scripts/ or ../scripts/ from operator/)
REPO_ROOT = Path(__file__).resolve().parent.parent


def process_rbac() -> bool:
    """Process RBAC manifest from kubebuilder output."""
    rbac_file = REPO_ROOT / "operator/config/rbac/role.yaml"
    output_file = REPO_ROOT / "charts/llama-agents/templates/rbac.yaml"

    if not rbac_file.exists():
        print(f"Error: {rbac_file} not found")
        return False

    # Read generated RBAC
    with open(rbac_file) as f:
        rbac = yaml.safe_load(f)

    if rbac.get("kind") != "ClusterRole":
        print(f"Error: Expected ClusterRole, got {rbac.get('kind')}")
        return False

    # Create Helm template with conditional and proper metadata.
    # Split mode (apps namespace != release namespace): apps-ns Role carries
    # every rule except leader-election leases, release-ns Role carries only
    # the leases rule (controller-runtime places the Lease in the operator
    # pod's own namespace). Single-namespace mode collapses to one Role.
    template = """{{- if .Values.rbac.create }}
{{- $appsNs := include "llama-agents.apps.namespace" . -}}
{{- $releaseNs := .Release.Namespace -}}
{{- $split := include "llama-agents.apps.splitNamespace" . -}}
{{- $saName := include "llama-agents.serviceAccountName" . -}}
{{- /*
Split mode (apps != release): apps-ns Role with everything except leases,
release-ns Role with only leases (leader-election Lease lives in the
operator pod's namespace). Single-namespace mode: one combined Role.
*/ -}}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: {{ $saName }}
  namespace: {{ $appsNs }}
  {{- with .Values.rbac.roleAnnotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
rules:
"""

    # Add the generated rules. The leader-election leases rule is conditional:
    # in split mode it moves to a separate release-ns Role below.
    for rule in rbac.get("rules", []):
        api_groups = rule.get("apiGroups", [])
        resources = rule.get("resources", [])
        verbs = rule.get("verbs", [])

        is_leases = api_groups == ["coordination.k8s.io"] and resources == ["leases"]
        if is_leases:
            template += "{{- if not $split }}\n"
        template += f"- apiGroups: {api_groups}\n"
        template += f"  resources: {resources}\n"
        template += f"  verbs: {verbs}\n"
        if is_leases:
            template += "{{- end }}\n"

    # Add RoleBinding (namespace-scoped) instead of ClusterRoleBinding
    template += """---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: {{ $saName }}
  namespace: {{ $appsNs }}
  {{- with .Values.rbac.roleBindingAnnotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: {{ $saName }}
subjects:
- kind: ServiceAccount
  name: {{ $saName }}
  namespace: {{ $releaseNs }}
{{- if $split }}
---
# Leader-election Lease lives in the operator pod's namespace.
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: {{ $saName }}-leader-election
  namespace: {{ $releaseNs }}
  {{- with .Values.rbac.roleAnnotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
rules:
- apiGroups: ['coordination.k8s.io']
  resources: ['leases']
  verbs: ['create', 'delete', 'get', 'list', 'patch', 'update', 'watch']
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: {{ $saName }}-leader-election
  namespace: {{ $releaseNs }}
  {{- with .Values.rbac.roleBindingAnnotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: {{ $saName }}-leader-election
subjects:
- kind: ServiceAccount
  name: {{ $saName }}
  namespace: {{ $releaseNs }}
{{- end }}
{{- end }}
"""

    # Write to Helm template
    with open(output_file, "w") as f:
        f.write(template)

    print(f"✅ Generated RBAC template with {len(rbac.get('rules', []))} rules")
    return True


def process_crds() -> bool:
    """Process CRD manifests from kubebuilder output."""
    crd_files = [
        REPO_ROOT
        / "operator/config/crd/bases/deploy.llamaindex.ai_llamadeployments.yaml",
        REPO_ROOT
        / "operator/config/crd/bases/deploy.llamaindex.ai_llamadeploymenttemplates.yaml",
    ]

    # Output 1: Raw CRD files in main chart's crds/ directory (install-only, no templating)
    crds_dir = REPO_ROOT / "charts/llama-agents/crds"
    crds_dir.mkdir(exist_ok=True)

    # Output 2: Raw CRD files in CRD chart's files/ directory (Helm template adds annotations)
    crd_chart_files_dir = REPO_ROOT / "charts/llama-agents-crds/files"
    crd_chart_files_dir.mkdir(parents=True, exist_ok=True)

    for crd_file in crd_files:
        if not crd_file.exists():
            print(f"Error: {crd_file} not found")
            return False
        content = crd_file.read_text()

        # Copy raw CRD to both destinations
        (crds_dir / crd_file.name).write_text(content)
        (crd_chart_files_dir / crd_file.name).write_text(content)

    print("Generated CRD files in crds/ directory (install-only)")
    print("Generated CRD files in llama-agents-crds/files/ (for CRD chart)")
    return True


def main() -> None:
    """Main entry point."""
    print("Processing kubebuilder-generated manifests...")

    success = True

    # Process RBAC
    if not process_rbac():
        success = False

    # Process CRDs
    if not process_crds():
        success = False

    if success:
        print("✅ All manifests processed successfully")
    else:
        print("❌ Some manifests failed to process")
        sys.exit(1)


if __name__ == "__main__":
    main()
