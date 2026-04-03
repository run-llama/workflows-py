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
