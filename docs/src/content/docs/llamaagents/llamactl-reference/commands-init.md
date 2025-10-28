---
title: init
sidebar:
  order: 101
---
Create a new app from a starter template, or update an existing app to the latest template version.

## Usage

```bash
llamactl init [--template <id>] [--dir <path>] [--force]
llamactl init --update
```

## Templates

- basic-ui: A basic starter workflow with a React Vite UI
- extraction-review: Extraction Agent with Review UI (Llama Cloud integration; review/correct extracted results)

If omitted, you will be prompted to choose interactively.

## Options

- `--update`: Update the current app to the latest template version. Ignores other options.
- `--template <id>`: Template to use (`basic-ui`, `extraction-review`).
- `--dir <path>`: Directory to create the new app in. Defaults to the template name.
- `--force`: Overwrite the directory if it already exists.

## What it does

- Copies the selected template into the target directory using [`copier`](https://copier.readthedocs.io/en/stable/)
- Adds assistant docs: `AGENTS.md` and symlinks `CLAUDE.md`/`GEMINI.md`
- initializes a Git repository if `git` is available
- Prints next steps to run locally and deploy

## Examples

- Interactive flow:
```bash
llamactl init
```

- Non‑interactive creation:
```bash
llamactl init --template basic-ui --dir my-app
```

- Overwrite an existing directory:
```bash
llamactl init --template basic-ui --dir ./basic-ui --force
```

- Update an existing app to the latest template:
```bash
llamactl init --update
```

See also: [Getting Started guide](/python/cloud/llamaagents/getting-started).
