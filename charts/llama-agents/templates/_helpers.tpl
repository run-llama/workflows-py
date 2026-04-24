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

{{/* s3proxy ConfigMap/Secret name (shared by both resources) */}}
{{- define "llama-agents.s3proxy.name" -}}
llama-agents-s3proxy
{{- end -}}

{{/* Chart-rendered Secret holding inline control plane S3 creds */}}
{{- define "llama-agents.controlplane.s3secret.name" -}}
llama-agents-controlplane-s3
{{- end -}}

{{/* Service account name — use value from values.yaml if set, otherwise chart name */}}
{{- define "llama-agents.serviceAccountName" -}}
{{- if .Values.serviceAccount.name -}}
{{ .Values.serviceAccount.name }}
{{- else -}}
{{ include "llama-agents.name" . }}
{{- end -}}
{{- end -}}

{{/* Namespace for LlamaDeployment CRs and child resources. Defaults to release namespace. */}}
{{- define "llama-agents.apps.namespace" -}}
{{- if and .Values.apps .Values.apps.namespace -}}
{{ .Values.apps.namespace }}
{{- else -}}
{{ .Release.Namespace }}
{{- end -}}
{{- end -}}

{{/* True when apps namespace differs from release namespace. */}}
{{- define "llama-agents.apps.splitNamespace" -}}
{{- if ne (include "llama-agents.apps.namespace" .) .Release.Namespace -}}
true
{{- end -}}
{{- end -}}
