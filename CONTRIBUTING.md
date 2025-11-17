# Contributing

## Ideas for Contributing

Contributions are welcome! We do our best to keep an eye on the issues and PRs and keeping them moving. Never hesitate to open a PR or an issue!

Generally, we're looking for:

- **New workflow features**: Improvements to the core workflow engine, step decorators, event handling, or context management
- **Better error handling and validation**: Enhanced error messages, better validation of workflow configurations, improved debugging capabilities, and better retry mechanisms
- **Performance optimizations**: Improvements to workflow execution speed, memory usage, or concurrent processing
- **Documentation improvements**: Better examples, tutorials, API documentation, or architectural explanations
- **Testing enhancements**: More comprehensive test coverage, performance tests, or integration tests
- **Checkpointing and persistence**: Improvements to workflow state management and resumption capabilities
- **Resource management**: Better handling of external resources and dependencies
- **Bug fixes**: Any issues you've encountered while using the library

Ideas beyond the above are welcome! This list is not exhaustive.

## Setup

This section assumes you have `uv` installed.

When developing locally, development works best with a virtual environment. You can create one with:

```bash
uv venv
# On MacOS/Linux
source .venv/bin/activate
# On Windows
.venv\Scripts\activate
```

The project is a monorepo with multiple closely related packages. You can install all dependencies with:

```bash
uv sync --all-extras --all-packages
```

The `pyproject.toml` files contain the dependencies for each project along with other details like the package name, version, etc. Generally you won't have to edit this file.

Linting is done automatically with `pre-commit`. Initialize it with:

```bash
uv run pre-commit install
```

## Run tests

Tests are run with `pytest`. You can run them with:

```bash
uv run pytest
```

Generally, all features should be covered by robust tests. If you are adding a new feature or fixing a bug, please add tests for it.

### Manually running linting

We use `pre-commit` to run linting and formatting on the codebase. You can run it manually with:

```bash
uv run pre-commit run -a
```

The `pre-commit` config is located in the `.pre-commit-config.yaml` file.

## Changesets and Releases

Despite being a Python project, we use [Changesets](https://github.com/changesets/changesets) for version management because it provides an excellent workflow for managing releases in a monorepo. Changesets makes it easy to track which packages need version bumps and helps generate changelogs automatically.

### Adding a Changeset

When you make a change that should be included in the next release, you need to add a changeset by running the following command (requires Node.js):

```bash
npx @changesets/cli
```

This will prompt you to:
1. Select which packages are affected by your changes
2. Choose the version bump type (major, minor, or patch)
3. Write a summary of your changes

### How It Works

When a PR with changesets is merged, the changeset bot will:
1. Sync versions from `package.json` to `pyproject.toml` files
2. Update package versions according to the changesets
3. Generate/update CHANGELOG files
4. Create a "Version Packages" PR that can be merged to trigger the release
