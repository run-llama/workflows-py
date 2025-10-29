---
title: pkg
sidebar:
  order: 106
---

:::caution
This command is currently limited to **agent workflows**.
Frontend packaging is **not yet supported**.
:::

The `pkg` command lets you package your application in different formats.
Currently supported formats: **Docker** and **Podman**.

It generates minimal, customizable build files from a deployment configuration, enabling you to easily package and deploy your application using container platforms.

For details on configuring your deployment file, see the [Deployment Config Reference](/python/cloud/llamaagents/configuration-reference).


## Usage

```bash
llamactl pkg [COMMAND] [options]
```

### Available Commands

- `container` – Generate a minimal, build-ready container file (e.g., `Dockerfile`) for your workflows using Docker or Podman.


## Command: `container`

```bash
llamactl pkg container [DEPLOYMENT_FILE] [options]
```

Generates a container build file compatible with **Docker** and **Podman** from a given deployment file and user-specified options.

### Options


- `deployment_file` - Path to the deployment file. Defaults to the current directory (`.`)
- `--python-version` - Python version for the base image. If not specified, it's inferred from `.python-version` or `pyproject.toml`; defaults to **3.12** if none found.
- `--port <int>` - Port to expose for the API server. Defaults to 4501.
- `--dockerignore-path` - Path for the generated `.dockerignore` file. Defaults to `.dockerignore`.
- `--overwrite` - Overwrite any existing output files. No default value.
- `--exclude <path>` - Path(s) to exclude from the build (appended to `.dockerignore`). Can be used multiple times. No default value.
- `--output-file` - Path and filename for the generated container build file. Defaults to `Dockerfile`.
- `--help` - Show help for this command and exit. No default value.

### Notes

- The generated `.dockerignore` file automatically excludes common Python-related directories such as:
  - Local virtual environments
  - Caches
  - Files that may contain sensitive data (e.g., `.env`)
- The produced container file is **minimal by design**. It should work for most cases, but you may need to customize it for specific use cases.

### Examples

**1. Create default `Dockerfile` and `.dockerignore`:**
```bash
llamactl pkg container
```

**2. Generate a custom container file with specific name, Python version, and port:**
```bash
llamactl pkg container --output-file Containerfile --python-version 3.14 --port 4502
```

**3. Exclude certain files or directories from the build:**
```bash
llamactl pkg container --exclude .env.local --exclude .github/workflows/ --exclude "*.pdf"
```
