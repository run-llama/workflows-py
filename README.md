# LlamaIndex Workflows Workspace

This repository now follows a packages-first layout. The previously root-level
`llama-index-workflows` library lives under `packages/llama-index-workflows`
alongside other related packages:

- `packages/llama-index-workflows` – the main runtime library
- `packages/llama-index-utils-workflow` – optional utilities that depend on the main library
- `packages/workflows-dev` – internal tooling shared by the workspace and CI

Each package contains its own `pyproject.toml`, dependencies, and test suite.
Run development commands from the relevant package directory, for example:

```bash
uv run --directory packages/llama-index-workflows pytest
uv run --directory packages/llama-index-workflows pre-commit run -a
```

The original product documentation, quick start guide, and examples now live in
`packages/llama-index-workflows/README.md`.
