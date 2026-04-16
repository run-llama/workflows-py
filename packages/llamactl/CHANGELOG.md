# llamactl

## 0.7.1

### Patch Changes

- facbac4: `PUBLIC_*` env var overlay for UI builds: `PUBLIC_X` overrides `X` in the build env so backend and frontend can use different URLs for the same service. Removes dead `VITE_`/`NEXT_PUBLIC_` injection from `llamactl serve`. Helm network policy gains `extraEgressRules`, DNS selector overrides, and `blockPrivateRanges` toggle.
- Updated dependencies [facbac4]
  - llama-agents-appserver@0.11.0

## 0.7.0

### Minor Changes

- e8b8f47: feat: add support for organizations
- e08c17c: Add shell tab-completion support with `llamactl completion generate` and `llamactl completion install`

### Patch Changes

- Updated dependencies [e8b8f47]
  - llama-agents-core@0.9.0
  - llama-agents-appserver@0.10.5

## 0.6.9

### Patch Changes

- 7ad3049: Reduce full clones from github for config, repo validation, and sha discovery. Reduce dependencies on system git, preferring dulwich
- Updated dependencies [7ad3049]
  - llama-agents-appserver@0.10.4
  - llama-agents-core@0.8.5

## 0.6.8

### Patch Changes

- Updated dependencies [286c91a]
  - llama-agents-appserver@0.10.3

## 0.6.7

### Patch Changes

- 740ee9e: Add a grace window to build artifact GC (configurable via `BUILD_ARTIFACT_GC_GRACE_SECONDS`, default 75m) and parallelize its delete loop with bounded concurrency. `llamactl auth`'s non-idempotent key-creation POST now only retries on connect-phase errors (`ConnectError`, `ConnectTimeout`, `PoolTimeout`) so initial-connectivity blips are absorbed without risking duplicate keys from a read-timeout retry.

## 0.6.6

### Patch Changes

- Updated dependencies [f27d98f]
  - llama-agents-core@0.8.4
  - llama-agents-appserver@0.10.2

## 0.6.5

### Patch Changes

- Updated dependencies [3f12660]
  - llama-agents-core@0.8.3
  - llama-agents-appserver@0.10.1

## 0.6.4

### Patch Changes

- Updated dependencies [3e2e7b8]
  - llama-agents-appserver@0.10.0

## 0.6.3

### Patch Changes

- 46f2675: security patches
- Updated dependencies [46f2675]
  - llama-agents-core@0.8.2
  - llama-agents-appserver@0.9.1

## 0.6.2

### Patch Changes

- Updated dependencies [58e7942]
  - llama-agents-appserver@0.9.0
  - llama-agents-core@0.8.1

## 0.6.1

### Patch Changes

- 68b1ec5: Use sqlite in agentcore, add local mode

## 0.6.0

### Minor Changes

- e2f3abd: Rename deployment name to display_name, add optional explicit id on create

### Patch Changes

- Updated dependencies [e2f3abd]
  - llama-agents-core@0.8.0
  - llama-agents-appserver@0.8.1

## 0.5.3

### Patch Changes

- llama-agents-appserver@0.8.0

## 0.5.2

### Patch Changes

- llama-agents-appserver@0.7.2

## 0.5.1

### Patch Changes

- Updated dependencies [7bb9a90]
  - llama-agents-core@0.7.0
  - llama-agents-appserver@0.7.1

## 0.5.0

### Minor Changes

- 9641415: Add dulwich-based git serving for internal repos. Users can push code via `llamactl push` and build pods clone via the build API. Bare repos are stored as tarballs in S3.

### Patch Changes

- llama-agents-appserver@0.7.0

## 0.4.26

### Patch Changes

- llama-agents-appserver@0.6.5

## 0.4.25

### Patch Changes

- llama-agents-appserver@0.6.4

## 0.4.24

### Patch Changes

- a15f1b4: Rename `llama_index_docs` MCP server identifier to `llama-index-docs` in scaffold config files
- Updated dependencies [4127101]
- Updated dependencies [1594315]
  - llama-agents-appserver@0.6.3

## 0.4.23

### Patch Changes

- 508b5da: Fix deployment update, fix github user auth
- Updated dependencies [508b5da]
  - llama-agents-core@0.6.2
  - llama-agents-appserver@0.6.2

## 0.4.22

### Patch Changes

- 32283aa: Replace async doc fetching with MCP server config generation
- Updated dependencies [1b86f90]
  - llama-agents-core@0.6.1
  - llama-agents-appserver@0.6.1

## 0.4.21

### Patch Changes

- Updated dependencies [4ab011f]
  - llama-agents-core@0.6.0
  - llama-agents-appserver@0.6.0

## 0.4.20

### Patch Changes

- Updated dependencies [eee29c1]
  - llama-deploy-appserver@0.5.3

## 0.4.19

### Patch Changes

- Updated dependencies [e11ad55]
  - llama-deploy-appserver@0.5.2

## 0.4.18

### Patch Changes

- llama-deploy-appserver@0.5.1

## 0.4.17

### Patch Changes

- 5588b7e: Bump to be compatible with latest appserver

## 0.4.16

### Patch Changes

- Updated dependencies [ac74af4]
- Updated dependencies [4ba0d9d]
  - llama-deploy-appserver@0.5.0
  - llama-deploy-core@0.5.0
