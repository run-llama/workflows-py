{{/*
envFrom helpers. Each emits zero-or-one `- secretRef:` entries with silent
BYO-over-inline precedence. Model: llamacloud's `_secrets.tpl`.
*/}}

{{/*
s3proxy sidecar: user-supplied `.secret` wins over chart-rendered Secret.
Emits nothing when neither is set (sidecar boots without creds).
*/}}
{{- define "llama-agents.secrets.s3proxy" -}}
{{- if .Values.s3proxy.secret }}
- secretRef:
    name: {{ .Values.s3proxy.secret }}
{{- else if .Values.s3proxy.config }}
- secretRef:
    name: {{ include "llama-agents.s3proxy.name" . }}
{{- end }}
{{- end -}}

{{/*
Control plane S3 creds. Precedence: s3.secret > outer secretRef (legacy alias)
> chart-rendered-from-inline. Emits nothing when all three are unset.
*/}}
{{- define "llama-agents.secrets.controlplaneS3" -}}
{{- $os := .Values.controlPlane.objectStorage }}
{{- if $os.s3.secret }}
- secretRef:
    name: {{ $os.s3.secret }}
{{- else if $os.secretRef }}
- secretRef:
    name: {{ $os.secretRef }}
{{- else if and $os.s3.accessKey $os.s3.secretKey }}
- secretRef:
    name: {{ include "llama-agents.controlplane.s3secret.name" . }}
{{- end }}
{{- end -}}
