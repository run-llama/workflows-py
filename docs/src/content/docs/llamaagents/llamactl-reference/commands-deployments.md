---
title: deployments
sidebar:
  order: 103
---
Deploy your app to the cloud and manage existing deployments. These commands operate on the project configured in your profile.

## Usage

```bash
llamactl deployments [COMMAND] [options]
```

Commands:

- `list`: List deployments for the configured project
- `get [DEPLOYMENT_ID] [--non-interactive]`: Show details; opens a live monitor unless `--non-interactive`
- `create`: Interactively create a new deployment
- `edit [DEPLOYMENT_ID]`: Interactively edit a deployment
- `delete [DEPLOYMENT_ID] [--confirm]`: Delete a deployment; `--confirm` skips the prompt
- `update [DEPLOYMENT_ID]`: Pull latest code from the configured branch and redeploy

Notes:

- If `DEPLOYMENT_ID` is omitted, you’ll be prompted to select one.
- All commands accept global options (profile, host, etc.).

## Commands

### List

```bash
llamactl deployments list
```

Shows a table of deployments with name, id, status, repo, deployment file, git ref, and secrets summary.

### Get

```bash
llamactl deployments get [DEPLOYMENT_ID] [--non-interactive]
```

- Default behavior opens a live monitor with status and streaming logs.
- Use `--non-interactive` to print details to the console instead of opening the monitor.

### Create (interactive)

```bash
llamactl deployments create
```

Starts an interactive flow to create a deployment. You can provide values like repository, branch, deployment file path, and secrets. (Flags such as `--repo-url`, `--name`, `--deployment-file-path`, `--git-ref`, `--personal-access-token` exist but creation is currently interactive.)

### Edit (interactive)

```bash
llamactl deployments edit [DEPLOYMENT_ID]
```

Opens an interactive form to update deployment settings.

### Delete

```bash
llamactl deployments delete [DEPLOYMENT_ID] [--confirm]
```

Deletes a deployment. Without `--confirm`, you’ll be asked to confirm.

### Update

```bash
llamactl deployments update [DEPLOYMENT_ID]
```

Refreshes the deployment to the latest commit on the configured branch and shows the resulting Git SHA change.

## See also

- Getting started: [Introduction](/python/cloud/llamaagents/getting-started)
- Configure names, env, and UI: [Deployment Config Reference](/python/cloud/llamaagents/configuration-reference)
- Local dev server: [`llamactl serve`](/python/cloud/llamaagents/llamactl-reference/commands-serve)
