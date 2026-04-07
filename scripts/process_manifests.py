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

    # Create Helm template with conditional and proper metadata
    # Generate Role (namespace-scoped) instead of ClusterRole
    template = """{{- if .Values.rbac.create }}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: {{ include "llama-agents.serviceAccountName" . }}
  namespace: {{ .Release.Namespace }}
  {{- with .Values.rbac.roleAnnotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
rules:
"""

    # Add the generated rules
    for rule in rbac.get("rules", []):
        api_groups = rule.get("apiGroups", [])
        resources = rule.get("resources", [])
        verbs = rule.get("verbs", [])

        template += f"- apiGroups: {api_groups}\n"
        template += f"  resources: {resources}\n"
        template += f"  verbs: {verbs}\n"

    # Add RoleBinding (namespace-scoped) instead of ClusterRoleBinding
    template += """---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: {{ include "llama-agents.serviceAccountName" . }}
  namespace: {{ .Release.Namespace }}
  {{- with .Values.rbac.roleBindingAnnotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: {{ include "llama-agents.serviceAccountName" . }}
subjects:
- kind: ServiceAccount
  name: {{ include "llama-agents.serviceAccountName" . }}
  namespace: {{ .Release.Namespace }}
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
