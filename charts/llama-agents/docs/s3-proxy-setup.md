# Non-S3 object storage

The chart can run [s3proxy](https://github.com/gaul/s3proxy) as a sidecar alongside the control plane, translating S3 API calls into the native API of your cloud provider.

When `s3proxy.enabled=true`:

- A ConfigMap (`llama-agents-s3proxy`) is always rendered with the sidecar's non-secret config.
- Credentials come from one of two sources — see below.
- The control plane pod gains a second container (`s3proxy`) and an `emptyDir` volume at `/tmp`.
- `S3_ENDPOINT_URL` defaults to `http://localhost:<containerPort>` and `S3_UNSIGNED` defaults to `true`. Explicit values still win.

The sidecar listens only on localhost — no Service, no cross-pod traffic, no NetworkPolicy changes.

## Inline config (chart renders the Secret)

Drop the `JCLOUDS_*` env vars for your provider from the [s3proxy storage-backend examples](https://github.com/gaul/s3proxy/wiki/Storage-backend-examples) into `s3proxy.config`:

```yaml
controlPlane:
  objectStorage:
    s3:
      bucket: my-bucket

s3proxy:
  enabled: true
  config:
    JCLOUDS_PROVIDER: <provider>
    JCLOUDS_IDENTITY: <id>
    JCLOUDS_CREDENTIAL: <secret>
    # ...any other JCLOUDS_* vars your backend needs
```

The chart renders a Secret named `llama-agents-s3proxy` with those keys and mounts it as `envFrom` on the sidecar.

## BYO Secret

Point `s3proxy.secret` at an existing Secret whose keys are the sidecar env vars:

```yaml
s3proxy:
  enabled: true
  secret: my-existing-s3proxy-secret
```

No chart Secret is rendered. The sidecar envFroms your Secret directly. Useful when credentials are managed out-of-band (Sealed Secrets, External Secrets, manual provisioning, etc.).

If both `config` and `secret` are set, `secret` wins silently.

## Gotchas

- **Neither `config` nor `secret` set**: the sidecar boots without credentials and every S3 call fails.
- **Explicit `controlPlane.objectStorage.s3.endpointUrl` overrides the sidecar**: the override wins and the sidecar runs unused.
- **Air-gapped clusters**: the default image is `docker.io/andrewgaul/s3proxy`. Mirror it and set `s3proxy.image`.
