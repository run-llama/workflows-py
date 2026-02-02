# LlamaIndex Workflows - Claude Development Guide

## Project Overview
This is the LlamaIndex Workflows library - an event-driven, async-first framework for orchestrating complex AI applications and multi-step processes.

## Key Technologies
- Python 3.9+
- AsyncIO (async/await)
- Pydantic for data models
- Starlette for web server
- Uvicorn for ASGI serving

## Development Commands

### Testing
```bash
# Run all tests
uv run --directory packages/llama-index-workflows pytest

# Run tests with coverage
uv run --directory packages/llama-index-workflows pytest --cov=src/workflows --cov-report=html

# Run specific test files
uv run --directory packages/llama-index-workflows pytest tests/test_server.py tests/test_server_utils.py

# Run tests in verbose mode
uv run --directory packages/llama-index-workflows pytest -v
```

### Linting & Formatting
```bash
# Run pre-commit hooks
uv run --directory packages/llama-index-workflows pre-commit run -a
```

## Project Structure
- `packages/llama-index-workflows/src/workflows/` - Main library code
- `packages/llama-index-workflows/src/workflows/server/` - Web server implementation
- `packages/llama-index-workflows/tests/` - Test suite
- `examples/` - Usage examples

## Key Components
- **Workflow** - Main orchestration class
- **Context** - State management across workflow steps
- **Events** - Event-driven communication between steps
- **WorkflowServer** - HTTP server for serving workflows as web services

## Notes for Claude
- Always run tests after making changes: `uv run --directory packages/llama-index-workflows pytest`
- Never use classes for tests, only use pytest functions
- Always annotate with types function arguments and return values
- The project uses async/await extensively
- Context serialization requires specific JSON format for globals

## Autonomous Operation

The following rules apply if you are running in an isolated sandbox environment and have tools to commit and push changes to git

Make sure to install uv as the package manager. Development commands rely on it.

```bash
curl -fsSL https://astral.sh/uv/install.sh | sh
```

Always run test tests and pre-commit commands before committing. They run very fast and are not verbose.

Tests:

```bash
uv run --directory packages/llama-index-workflows pytest -nauto --timeout=1
```

Linting, typechecking, and formatting:
```bash
uv run --directory packages/llama-index-workflows pre-commit run -a
```

## Testing Patterns

We use **pytest** with idiomatic pytest patterns. Follow these guidelines:

- **No Test Classes**: Do not use test classes to organize tests. Write tests as standalone functions. Achieve organization through descriptive function names (e.g., `test_create_job_with_invalid_input_raises_error`) or by splitting into separate test files.
- **Pytest Fixtures**: Use fixtures for setup/teardown and shared test dependencies. Prefer fixtures over manual setup code repeated across tests.
- **Prefer Real Objects Over Mocks**: Use simple dataclasses and real objects directly when available rather than mocking them. Only mock external dependencies or things that are truly difficult to instantiate.
- **DRY Test Setup**: Do not repeat patches or setup code. Create reusable abstractions—fixtures, helper functions, or module-level constants—that can be shared across tests. Tests can easily be overwhelmed with setup; start from a rich suite of testing utilities to enable many small, expressive tests.
- **Simple Testing Utilities**: Testing utilities should be basic—just functions, fixtures, and global variables. Avoid over-engineering test infrastructure.

## Coding Style

- Always use `from __future__ import annotations` at the top of each test file. Never use string annotations.
- Include the standard SPDX license header at the top of each test file.
- Comments are useful, but avoid fluff.
- Never use inline imports unless required to prevent circular dependencies.
