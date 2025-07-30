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
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/workflows --cov-report=html

# Run specific test files
uv run pytest tests/test_server.py tests/test_server_utils.py

# Run tests in verbose mode
uv run pytest -v
```

### Linting & Formatting
```bash
# Run pre-commit hooks
uv run -- pre-commit run -a
```

## Project Structure
- `src/workflows/` - Main library code
- `src/workflows/server/` - Web server implementation
- `tests/` - Test suite
- `examples/` - Usage examples

## Key Components
- **Workflow** - Main orchestration class
- **Context** - State management across workflow steps
- **Events** - Event-driven communication between steps
- **WorkflowServer** - HTTP server for serving workflows as web services

## Notes for Claude
- Always run tests after making changes: `uv run pytest`
- Never use classes for tests, only use pytest functions
- Always annotate with types function arguments and return values
- The project uses async/await extensively
- Context serialization requires specific JSON format for globals
