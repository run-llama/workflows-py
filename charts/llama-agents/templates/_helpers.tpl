{{/*
Common name helpers for the llama-agents chart.
All resource names derive from these so a single rename propagates everywhere.
*/}}

{{/* Chart name */}}
{{- define "llama-agents.name" -}}
llama-agents
{{- end -}}

{{/* Control plane deployment and related resources */}}
{{- define "llama-agents.controlplane.name" -}}
llama-agents-control-plane
{{- end -}}

{{/* Operator deployment and related resources */}}
{{- define "llama-agents.operator.name" -}}
llama-agents-operator
{{- end -}}

{{/* Main API service */}}
{{- define "llama-agents.service.name" -}}
llama-agents-service
{{- end -}}

{{/* Build API service */}}
{{- define "llama-agents.build.name" -}}
llama-agents-build
{{- end -}}

{{/* Service account name — use value from values.yaml if set, otherwise chart name */}}
{{- define "llama-agents.serviceAccountName" -}}
{{- if .Values.serviceAccount.name -}}
{{ .Values.serviceAccount.name }}
{{- else -}}
{{ include "llama-agents.name" . }}
{{- end -}}
{{- end -}}

{{/*
Apps namespace — the namespace where LlamaDeployment CRs and their child
resources live. Defaults to the release namespace (legacy single-namespace
mode). Setting .Values.operator.apps.namespace targets a separate namespace
for apps while keeping the operator + control plane in the release namespace.
*/}}
{{- define "llama-agents.apps.namespace" -}}
{{- if and .Values.operator.apps .Values.operator.apps.namespace -}}
{{ .Values.operator.apps.namespace }}
{{- else -}}
{{ .Release.Namespace }}
{{- end -}}
{{- end -}}

{{/*
True when the apps namespace differs from the release namespace. Used to
decide whether to emit split RBAC and cross-namespace NetworkPolicy selectors.
*/}}
{{- define "llama-agents.apps.splitNamespace" -}}
{{- if ne (include "llama-agents.apps.namespace" .) .Release.Namespace -}}
true
{{- end -}}
{{- end -}}
