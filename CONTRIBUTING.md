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

## Code Structure

Generally, the code organization is relatively flat. Here's a quick overview of the code structure:

- **[src/workflows/workflow.py](src/workflows/workflow.py)**: The main `Workflow` class that orchestrates step execution and event handling
- **[src/workflows/decorators.py](src/workflows/decorators.py)**: The `@step` decorator used to mark functions as workflow steps, along with step configuration
- **[src/workflows/events.py](src/workflows/events.py)**: Core event types (`StartEvent`, `StopEvent`, `Event`) that drive workflow execution
- **[src/workflows/handler.py](src/workflows/handler.py)**: The `WorkflowHandler` that manages workflow execution and provides the interface for running workflows
- **[src/workflows/context/](src/workflows/context/)**: Context management for workflow state and data passing between steps
  - **[src/workflows/context/context.py](src/workflows/context/context.py)**: Main `Context` class that holds workflow state and handles event routing
  - **[src/workflows/context/serializers.py](src/workflows/context/serializers.py)**: Serialization utilities for persisting workflow state
- **[src/workflows/resource.py](src/workflows/resource.py)**: Management of external resources that workflows can depend on
- **[src/workflows/retry_policy.py](src/workflows/retry_policy.py)**: Configurable retry logic for failed steps
- **[src/workflows/errors.py](src/workflows/errors.py)**: Custom exception types for workflow-specific errors
- **[src/workflows/types.py](src/workflows/types.py)**: Type definitions and type variables used throughout the library
- **[src/workflows/utils.py](src/workflows/utils.py)**: Utility functions for step introspection, signature validation, and other helpers

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

Then, install the dependencies with:

```bash
uv sync
```

The `pyproject.toml` file contains the dependencies for the project along with other details like the package name, version, etc. Generally you won't have to edit this file.

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
uv run -- pre-commit run -a
```

The `pre-commit` config is located in the `.pre-commit-config.yaml` file.

## Release (requires repo push access)

- Update the `version` key in the `project` table of the `pyproject.toml` file.
  - Change the version numbers following the [SemVer](https://semver.org/) specification.
- Commit directly to `main`
- Tag the commit on `main` you just created using an annotated tag: `git tag -m"v0.0.0" v0.0.0`
- Push the `main` branch along with tags: `git push origin main --tags`
- Monitor the [release workflow](https://github.com/run-llama/workflows-py/actions/workflows/publish_release.yml)
  - Check the [release page](https://github.com/run-llama/workflows-py/releases) contains the new changelog, manually amend it if necessary
  - Check the package was published on [PyPI](https://pypi.org/project/llama-index-workflows/)
