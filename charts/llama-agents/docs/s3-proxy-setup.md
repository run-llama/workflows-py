# Non-S3 object storage

The chart can run [s3proxy](https://github.com/gaul/s3proxy) as a sidecar alongside the control plane, translating S3 API calls into the native API of your cloud provider.

When `s3proxy.enabled=true`:

- A ConfigMap and Secret (`llama-agents-s3proxy`) are rendered. `s3proxy.config` is a raw passthrough to the Secret — keys become environment variables on the sidecar.
- The control plane pod gains a second container (`s3proxy`) and an `emptyDir` volume at `/tmp`.
- `S3_ENDPOINT_URL` defaults to `http://localhost:<containerPort>` and `S3_UNSIGNED` defaults to `true`. Explicit values still win.

The sidecar listens only on localhost — no Service, no cross-pod traffic, no NetworkPolicy changes.

## Configuring a backend

Pick the `JCLOUDS_*` env vars for your provider from the [s3proxy storage-backend examples](https://github.com/gaul/s3proxy/wiki/Storage-backend-examples) and drop them into `s3proxy.config`:

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

## Gotchas

- **Empty `s3proxy.config`**: s3proxy starts without a backend and every S3 call fails.
- **Explicit `controlPlane.objectStorage.s3.endpointUrl` overrides the sidecar**: the override wins and the sidecar runs unused.
- **Air-gapped clusters**: the default image is `docker.io/andrewgaul/s3proxy`. Mirror it and set `s3proxy.image`.
