---
title: auth
sidebar:
  order: 104
---
Authenticate and manage profiles for the current environment. Profiles store your control plane API URL, project, and optional API key.

## Usage

```bash
llamactl auth [COMMAND] [options]
```

Commands:

- `token [--project-id ID] [--api-key KEY] [--interactive/--no-interactive]`: Create profile from API key; validates token and selects a project
- `login`: Login via web browser (OIDC device flow) and create a profile
- `list`: List login profiles in the current environment
- `switch [NAME] [--interactive/--no-interactive]`: Set currently logged in user/token
- `logout [NAME] [--interactive/--no-interactive]`: Delete a login and its local data
- `project [PROJECT_ID] [--interactive/--no-interactive]`: Change the active project for the current profile

Notes:

- Profiles are filtered by the current environment (`llamactl auth env switch`).
- Non-interactive `token` requires both `--api-key` and `--project-id`.

## Commands

### Token

```bash
llamactl auth token [--project-id ID] [--api-key KEY] [--interactive/--no-interactive]
```

- Interactive: Prompts for API key (masked), validates it by listing projects, then lets you choose a project. Creates an auto‑named profile and sets it current.
- Non‑interactive: Requires both `--api-key` and `--project-id`.

### Login

```bash
llamactl auth login
```

Login via your browser using the OIDC device flow, select a project, and create a login profile set as current.

### List

```bash
llamactl auth list
```

Shows a table of profiles for the current environment with name and active project. The current profile is marked with `*`.

### Switch

```bash
llamactl auth switch [NAME] [--interactive/--no-interactive]
```

Set the current profile. If `NAME` is omitted in interactive mode, you will be prompted to select one.

### Logout

```bash
llamactl auth logout [NAME] [--interactive/--no-interactive]
```

Delete a profile. If the deleted profile is current, the current selection is cleared.

### Project

```bash
llamactl auth project [PROJECT_ID] [--interactive/--no-interactive]
```

Change the active project for the current profile. In interactive mode, select from server projects. In environments that don't require auth, you can also enter a project ID.

## See also

- Environments: [`llamactl auth env`](/python/cloud/llamaagents/llamactl-reference/commands-auth-env)
- Getting started: [Introduction](/python/cloud/llamaagents/getting-started)
- Deployments: [`llamactl deployments`](/python/cloud/llamaagents/llamactl-reference/commands-deployments)
