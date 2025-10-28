---
title: serve
sidebar:
  order: 102
---
Serve your app locally for development and testing. Reads configuration from your project (e.g., `pyproject.toml` or `llama_deploy.yaml`) and starts the Python API server, optionally proxying your UI in dev.

See also: [Deployment Config Reference](/python/cloud/llamaagents/configuration-reference) and [UI build and dev integration](/python/cloud/llamaagents/ui-build).

## Usage

```bash
llamactl serve [DEPLOYMENT_FILE] [options]
```

- `DEPLOYMENT_FILE` defaults to `.` (current directory). Provide a path to a specific deployment file or directory if needed.

## Options

- `--no-install`: Skip installing Python and JS dependencies
- `--no-reload`: Disable API server auto‑reload on code changes
- `--no-open-browser`: Do not open the browser automatically
- `--preview`: Build the UI to static files and serve them (production‑like)
- `--port <int>`: Port for the API server
- `--ui-port <int>`: Port for the UI proxy in dev

## Behavior

- Prepares the server environment (installs dependencies unless `--no-install`)
- In dev mode (default), proxies your UI dev server and reloads on change
- In preview mode, builds the UI to static files and serves them without a proxy
