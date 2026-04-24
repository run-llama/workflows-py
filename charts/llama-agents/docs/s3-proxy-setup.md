# Deploying llama-agents on non-AWS object storage

The chart can run [s3proxy](https://github.com/gaul/s3proxy) as a sidecar alongside the control plane, translating S3 API calls into the native API of your cloud provider. Flip one flag, fill in credentials, and the control plane talks to Azure Blob, GCS, or any [jclouds-supported backend](https://github.com/gaul/s3proxy/wiki/Storage-backend-examples).

## How it works

When `s3proxy.enabled=true`:

- A ConfigMap (`llama-agents-s3proxy`) is rendered with the static s3proxy settings.
- A Secret (`llama-agents-s3proxy`) is rendered from `s3proxy.config` — these are the `JCLOUDS_*` env vars that select and authenticate against your backend.
- The control plane pod gains a second container (`s3proxy`) and an `emptyDir` volume at `/tmp`.
- The control plane's `S3_ENDPOINT_URL` defaults to `http://localhost:<containerPort>` and `S3_UNSIGNED` defaults to `true`. Explicit user values still win.

The sidecar only listens on localhost — no Service, no cross-pod traffic, no NetworkPolicy changes.

## Azure Blob Storage

```yaml
controlPlane:
  objectStorage:
    s3:
      # The Azure container name to use as the S3 bucket.
      bucket: my-container

s3proxy:
  enabled: true
  config:
    JCLOUDS_PROVIDER: azureblob
    JCLOUDS_IDENTITY: <storage-account-name>
    JCLOUDS_CREDENTIAL: <storage-account-key>
    # Optional, defaults to https://<account>.blob.core.windows.net
    # JCLOUDS_ENDPOINT: https://<account>.blob.core.windows.net
```

Install:

```bash
helm install llama-agents oci://docker.io/llamaindex/llama-agents \
  -f my-values.yaml
```

## Google Cloud Storage

GCS requires an interoperability (HMAC) key on the target bucket.

```yaml
controlPlane:
  objectStorage:
    s3:
      bucket: my-gcs-bucket
      region: us-central1

s3proxy:
  enabled: true
  config:
    JCLOUDS_PROVIDER: google-cloud-storage
    JCLOUDS_IDENTITY: <hmac-access-id>
    JCLOUDS_CREDENTIAL: <hmac-secret>
```

## Bring-your-own credentials via `--set-string`

Secrets don't have to live in a values file:

```bash
helm install llama-agents oci://docker.io/llamaindex/llama-agents \
  --set controlPlane.objectStorage.s3.bucket=my-container \
  --set s3proxy.enabled=true \
  --set s3proxy.config.JCLOUDS_PROVIDER=azureblob \
  --set-string s3proxy.config.JCLOUDS_IDENTITY=$AZ_ACCOUNT \
  --set-string s3proxy.config.JCLOUDS_CREDENTIAL=$AZ_KEY
```

## Gotchas

- **Empty `s3proxy.config`**: s3proxy starts without a backend and every S3 call fails. Fill in at least `JCLOUDS_PROVIDER`, `JCLOUDS_IDENTITY`, and `JCLOUDS_CREDENTIAL`.
- **Explicit `controlPlane.objectStorage.s3.endpointUrl` overrides the sidecar**: The override wins and the sidecar runs unused. Unset `endpointUrl` if you want the default localhost wiring.
- **Each control plane replica runs its own sidecar**. That's fine — s3proxy is stateless and only reached from within its pod.
- **Air-gapped clusters**: the default image is `docker.io/andrewgaul/s3proxy`. Mirror it and set `s3proxy.image` to your copy.

## References

- [s3proxy storage-backend examples](https://github.com/gaul/s3proxy/wiki/Storage-backend-examples)
- [s3proxy Dockerfile env vars](https://github.com/gaul/s3proxy/blob/master/Dockerfile)
