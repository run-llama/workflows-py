---
title: Getting Started
sidebar:
  order: 1
---
:::caution
Cloud deployments of LlamaAgents is still in alpha. You can try it out locally, or [request access by contacting us](https://landing.llamaindex.ai/llamaagents?utm_source=docs)
:::

## Getting Started with `llamactl`

LlamaAgents uses the [`llamactl` CLI for development](https://pypi.org/project/llamactl/). `llamactl` bootstraps an application server that manages running and persisting your workflows, and a control plane for managing cloud deployments of applications. It has some system pre-requisites that must be installed in order to work:

- Make sure you have [`uv`](https://docs.astral.sh/uv/getting-started/installation/) installed. `uv` is a Python package manager and build tool. `llamactl` integrates with it in order to quickly manage your project's build and dependencies.
- Windows support is experimental (as of version `0.3.14`) and requires some adjustments to run `llamactl` without issues: see [our dedicated guide](https://github.com/run-llama/llamactl-windows) on the topic. For a better user experience, it is still advisable to use WSL2 (e.g., Ubuntu) and follow the Linux instructions. See [Install WSL](https://learn.microsoft.com/windows/wsl/install).
- Likewise, Node.js is required for UI development. For macOS and Linux, we recommend installing Node via [`nvm`](https://github.com/nvm-sh/nvm) to manage versions. You can use your node package manager of choice (`npm`, `pnpm`, or `yarn`). For Windows, we recommend using [Chocolatey](https://community.chocolatey.org/packages/nodejs) for the installation process.
- Ensure `git` is installed:
  - macOS: [Install via Xcode Command Line Tools](https://git-scm.com/download/mac) or Homebrew (`brew install git`)
  - Linux: Follow your distro instructions: [git-scm.com/download/linux](https://git-scm.com/download/linux)
  - Windows: use [Chocolatey](https://community.chocolatey.org/packages/git.install)

## Install

Choose one:

- Try without installing:
```bash
uvx llamactl --help
```

- Install globally (recommended):
```bash
uv tool install -U llamactl
llamactl --help
```

## Initialize a Project

`llamactl` includes starter templates for both full‑stack UI apps, and headless (API only) workflows. Pick a template and customize it.

:::warning
Since llamactl uses symlinks when initializing, you might run into some permission issues with Windows. We advise you to activate Developer Settings using `start ms-settings:developers` before running `llamactl init`.
:::

```bash
llamactl init
```

This will prompt for some details, and create a Python module that contains LlamaIndex workflows, plus an optional UI you can serve as a static frontend.

:::info
When you run `llamactl init`, the scaffold also includes AI assistant-facing docs: `AGENTS.md`, `CLAUDE.md`, and `GEMINI.md`. These contain quick references and instructions for using LlamaIndex libraries to assist coding. These files are optional and safe to customize or remove — they do not affect your builds, runtime, or deployments.
:::

Application configuration is managed within your project's `pyproject.toml`, where you can define Python workflow instances that should be served, environment details, and configuration for how the UI should be built. See the [Deployment Config Reference](/python/llamaagents/llamactl/configuration-reference) for details on all configurable fields.

## Develop and Run Locally

Once you have a project, you can run the dev server for your application:

```bash
llamactl serve
```

`llamactl serve` will

1. Install all required dependencies
2. Read the workflows configured in your app’s `pyproject.toml` and serve them as an API
3. Start up and proxy the frontend development server, so you can seamlessly write a full stack application.

For example, with the following configuration, the app will be served at `http://localhost:4501/deployments/my-package`. Make a `POST` request to `/deployments/my-package/workflows/my-workflow/run` to trigger the workflow in `src/my_package/my_workflow.py`.

```toml
[project]
name = "my-package"
# ...
[tool.llamadeploy.workflows]
my-workflow = "my_package.my_workflow:workflow"

[tool.llamadeploy.ui]
directory = "ui"
```

```py
# src/my_package/my_workflow.py
# from workflows import ...
# ...
workflow = MyWorkflow()
```

At this point, you can get to coding. The development server will detect changes as you save files. It will even resume in-progress workflows!

For more information about CLI flags available, see [`llamactl serve`](/python/llamaagents/llamactl-reference/commands-serve).

For a more detailed reference on how to define and expose workflows, see [Workflows & App Server API](/python/llamaagents/llamactl/workflow-api).

## Create a Cloud Deployment

LlamaAgents applications can be rapidly deployed just by pointing to a source git repository. With the provided repository configuration, LlamaCloud will clone, build, and serve your app. It can even access GitHub private repositories by installing the [GitHub app](https://github.com/apps/llama-deploy)

Example:

```bash
git remote add origin https://github.com/org/repo
git add -A
git commit -m 'Set up new app'
git push -u origin main
```

Then, create a deployment:

```bash
llamactl deployments create
```

:::info
The first time you run this, you'll be prompted to log into LlamaCloud.

Username/password sign-in is not yet supported. If you do not have a supported social sign-in provider, you can use token-based authentication via `llamactl auth token`. See [`llamactl auth`](/python/llamaagents/llamactl-reference/commands-auth) for details.
:::

This will open an interactive Terminal UI (TUI). You can tab through fields, or even point and click with your mouse if your terminal supports it. All required fields should be automatically detected from your environment, but can be customized:

- Name: Human‑readable and URL‑safe; appears in your deployment URL
- Git repository: Public HTTP or private GitHub (install the GitHub app for private repos)
- Git branch: Branch to pull and build from (use `llamactl deployments update` to roll forward). This can also be a tag or a git commit.
- Secrets: Pre‑filled from your local `.env`; edit as needed. These cannot be read again after creation.

When you save, LlamaAgents will verify that it has access to your repository (and prompt you to install the GitHub app if not)

After creation, the TUI will show deployment status and logs.
- You can later use `llamactl deployments get` to view again.
- You can add secrets or change branches with `llamactl deployments edit`.
- If you update your source repo, run `llamactl deployments update` to roll a new version.

---

Next: Read about defining and exposing workflows in [Workflows & App Server API](/python/llamaagents/llamactl/workflow-api).
