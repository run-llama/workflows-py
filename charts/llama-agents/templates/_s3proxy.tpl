{{/*
s3proxy sidecar container definition.
Rendered into the control plane pod when `.Values.s3proxy.enabled` is true.
*/}}
{{- define "llama-agents.s3proxy.container" -}}
- name: s3proxy
  image: {{ .Values.s3proxy.image | quote }}
  imagePullPolicy: {{ .Values.s3proxy.imagePullPolicy | default "IfNotPresent" }}
  {{- with .Values.s3proxy.securityContext }}
  securityContext:
    {{- toYaml . | nindent 4 }}
  {{- end }}
  ports:
  - name: s3proxy
    containerPort: {{ int .Values.s3proxy.containerPort }}
    protocol: TCP
  env:
  - name: LOG_LEVEL
    value: {{ .Values.s3proxy.logLevel | default "info" | quote }}
  - name: S3PROXY_LOG_LEVEL
    value: {{ .Values.s3proxy.logLevel | default "info" | quote }}
  envFrom:
  - configMapRef:
      name: {{ include "llama-agents.s3proxy.name" . }}
  {{- include "llama-agents.secrets.s3proxy" . | nindent 2 }}
  {{- with .Values.s3proxy.resources }}
  resources:
    {{- toYaml . | nindent 4 }}
  {{- end }}
  volumeMounts:
  - name: s3proxy-tmp
    mountPath: /tmp
    subPath: tmp-dir
{{- end -}}

{{/*
s3proxy ConfigMap data (non-secret config).
*/}}
{{- define "llama-agents.s3proxy.configMapData" -}}
S3PROXY_AUTHORIZATION: "none"
S3PROXY_CORS_ALLOW_ORIGINS: "*"
S3PROXY_ENDPOINT: {{ printf "http://0.0.0.0:%d" (int .Values.s3proxy.containerPort) | quote }}
S3PROXY_IGNORE_UNKNOWN_HEADERS: "true"
{{- end -}}

{{/*
s3proxy Secret data (passthrough of .Values.s3proxy.config, b64-encoded).
*/}}
{{- define "llama-agents.s3proxy.secretData" -}}
{{- range $key, $value := .Values.s3proxy.config }}
{{ $key }}: {{ $value | toString | b64enc | quote }}
{{- end }}
{{- end -}}

{{/*
Endpoint URL the control plane should use to reach the sidecar on localhost.
*/}}
{{- define "llama-agents.s3proxy.localEndpoint" -}}
{{- printf "http://localhost:%d" (int .Values.s3proxy.containerPort) -}}
{{- end -}}
